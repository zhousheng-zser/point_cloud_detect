import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ==== 单个目标跟踪器 ====
class Tracker:
    def __init__(self, bbox3d, id, timestamp ,init_speed=0 ):
        """
        bbox3d: [x, y, z, l, w, h, yaw,line]
        KF 只跟踪 [x, y, z, vx, vy, vz]
        """
        self.last_observation = np.array(bbox3d)
        self.id = id
        self.time_since_update = 0
        self.hits = 1
        self.speed = init_speed  # 只维护一个速度标量
        self.last_timestamp = timestamp  # 毫秒时间戳
        self.popped = False  # 默认没被取过

        # ========== 独立属性 ==========
        self.length = bbox3d[3]
        self.width  = bbox3d[4]
        self.height = bbox3d[5]
        self.yaw    = bbox3d[6]
        self.line   = bbox3d[7]

        # ==== Kalman Filter ====
        # 6维
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # 初始状态
        self.kf.x[:3] = np.array(bbox3d[:3]).reshape(3,1)   # 位置
        self.kf.P *= 10.0

        # 状态转移矩阵 F
        self.kf.F = np.eye(6)
        for i in range(3):
            self.kf.F[i, i+3] = 1.0  # 位置由速度更新

        # 观测矩阵 H
        H = np.zeros((3,6))
        H[0,0] = H[1,1] = H[2,2] = 1
        self.kf.H = H

        # 噪声
        q_pos = 1e-3
        q_vel = 1e-2
        Q = np.zeros((6,6))
        Q[0:3,0:3] = np.eye(3) * q_pos
        Q[3:6,3:6] = np.eye(3) * q_vel
        self.kf.Q = Q

        # 测量噪声 R
        self.kf.R = np.eye(3) * 0.25

    def predict(self, dt=0.1):
        for i in range(3):
            self.kf.F[i, i+3] = dt
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox3d, timestamp):
        bbox3d = np.array(bbox3d, dtype=float)
        # KF 更新位置
        self.kf.update(bbox3d[:3])

        #长宽高取max
        self.length = max(self.length, bbox3d[3])
        self.width  = max(self.width, bbox3d[4])
        self.height = max(self.height, bbox3d[5])
        self.yaw  = bbox3d[6]
        self.line = bbox3d[7]

        #更新速度标量
        dt = (timestamp - self.last_timestamp) / 1000.0  # ms -> s
        if dt > 1e-6:
            dist = np.linalg.norm(bbox3d[:3] - self.last_observation[:3])
            #print(f"new_speed={dist} / {dt} = {dist / dt}" )
            self.speed = dist / dt
        self.last_timestamp = timestamp

        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        """
        返回预测的 [x,y,z,l,w,h,yaw,line]
        """
        state = self.kf.x.flatten()
        return np.array([state[0], state[1], state[2],
        self.length,self.width,self.height,self.yaw,self.line
        ])
    def get_last_observation(self):
        """
        返回最近一次 update() 的观测值 [x,y,z,l,w,h,yaw,line]
        """
        return self.last_observation
    def get_speed(self):
        return self.speed

# ==== 多目标跟踪管理器 ====
class MultiObjectTracker:
    def __init__(self, dist_threshold=5, max_age=3):
        self.trackers = []
        self.next_id = 0
        self.dist_threshold = dist_threshold
        self.max_age = max_age

    def get_iou(self, x1, y1, z1, l1, w1, h1, yaw1, line1,
                      x2, y2, z2, l2, w2, h2, yaw2, line2):
        if line1 != line2:
            return 0.0

        # box1
        box1 = [h1, w1, l1, x1, y1, z1, yaw1]
        # box2
        box2 = [h2, w2, l2, x2, y2, z2, yaw2]

        return self.iou_3d_noshapely_fixed(box1, box2)

    def iou_3d_noshapely_fixed(self, box1, box2):
        """
        box = [h, w, l, x, y, z, ry]
        KITTI 格式: (x,y,z) 是底部中心
        """
        # 体积
        vol1 = box1[0] * box1[1] * box1[2]
        vol2 = box2[0] * box2[1] * box2[2]
        

        # BEV 多边形
        rect1 = self.get_corners_bev(box1)
        rect2 = self.get_corners_bev(box2)

        # 相交多边形 (Sutherland–Hodgman)
        inter_poly = self.convex_hull_intersection(rect1, rect2)
        if inter_poly is None:
            return 0.0
        inter_area = self.polygon_area(inter_poly)

        # 高度重叠
        y1_min, y1_max = box1[4], box1[4] + box1[0]
        y2_min, y2_max = box2[4], box2[4] + box2[0]
        inter_h = min(y1_max, y2_max) - max(y1_min, y2_min)
        if inter_h <= 0:
            return 0.0

        # 相交体积
        inter_vol = inter_area * inter_h
        union_vol = vol1 + vol2 - inter_vol
        if union_vol <= 0:
            return 0.0
        return inter_vol / union_vol

    def get_corners_bev(self, box):
        """KITTI box -> BEV 四边形"""
        h, w, l, x, y, z, ry = box
        R = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0,          1, 0],
            [-np.sin(ry),0, np.cos(ry)]
        ])
        # 3D 8 corners (局部坐标系, y=0在底面)
        x_corners = [l/2,  l/2, -l/2, -l/2, l/2,  l/2, -l/2, -l/2]
        y_corners = [0,0,0,0,h,h,h,h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners = np.vstack([x_corners, y_corners, z_corners])
        corners = R @ corners
        corners = corners + np.array([[x], [y], [z]])
        rect = corners[[0,2], :4].T

        # === 顺时针排序 ===
        cx, cy = rect[:,0].mean(), rect[:,1].mean()
        angles = np.arctan2(rect[:,1] - cy, rect[:,0] - cx)
        sort_idx = np.argsort(angles)
        rect_sorted = rect[sort_idx]

        return rect_sorted

    def polygon_area(self, poly):
        """多边形面积 (shoelace formula)"""
        x = poly[:,0]
        y = poly[:,1]
        return 0.5*np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

    def is_point_inside(self, p, poly):
        """判断点是否在凸多边形内 (射线法)"""
        x, y = p
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1)+x1):
                inside = not inside
        return inside

    def convex_hull_intersection(self, p1, p2):
        """两凸多边形相交 (Sutherland-Hodgman)"""
        def clip(subjectPolygon, clipPolygon):
            def inside(p, cp1, cp2):
                return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
            def computeIntersection(s, e, cp1, cp2):
                dc = [cp1[0]-cp2[0], cp1[1]-cp2[1]]
                dp = [s[0]-e[0], s[1]-e[1]]
                n1 = cp1[0]*cp2[1] - cp1[1]*cp2[0]
                n2 = s[0]*e[1] - s[1]*e[0]
                n3 = dc[0]*dp[1] - dc[1]*dp[0]
                if n3 == 0:
                    return [0,0]
                x = (n1*dp[0] - n2*dc[0]) / n3
                y = (n1*dp[1] - n2*dc[1]) / n3
                return [x, y]

            outputList = subjectPolygon
            cp1 = clipPolygon[-1]
            for cp2 in clipPolygon:
                inputList = outputList
                outputList = []
                if len(inputList) == 0:
                    return []
                s = inputList[-1]
                for e in inputList:
                    if inside(e, cp1, cp2):
                        if not inside(s, cp1, cp2):
                            outputList.append(computeIntersection(s, e, cp1, cp2))
                        outputList.append(e)
                    elif inside(s, cp1, cp2):
                        outputList.append(computeIntersection(s, e, cp1, cp2))
                    s = e
                cp1 = cp2
            return outputList

        inter_p = clip(p1.tolist(), p2.tolist())
        if len(inter_p) == 0:
            return None
        return np.array(inter_p)

    def euclidean_distance(self, b1, b2):
        """
        计算两个3D框的欧式距离 (你已有实现)
        b1, b2 = [x, y, z, l, w, h, yaw, line]
        """
        if b1[7]!= b2[7]:
            return 99999999
        dist = np.linalg.norm(np.array(b1[:3]) - np.array(b2[:3]))
        return dist

    def update(self, detections,timestamp):
        """
        detections: [[x,y,z,l,w,h,yaw,line], ...]
        """
        # Step 1: 预测
        predicted_states = [] 
        for idx, trk in enumerate(self.trackers):
            # === 用真实 dt 预测 ===
            dt = max(1e-3, (timestamp - trk.last_timestamp) / 1000.0)
            predicted_states.append(trk.predict(dt=dt))

        # Step 2: 构建代价矩阵 (欧氏距离)
        cost = np.zeros((len(self.trackers), len(detections)))
        for i, trk in enumerate(self.trackers):
            for j, det in enumerate(detections):
                dist = self.euclidean_distance(trk.get_state(), det)
                cost[i, j] = dist  #之后可以加个判断体积差

        # Step 3: 匹配    匹配上的, 没匹配上的, 要删的 
        matched, unmatched_trk, unmatched_det = [], [], []
        if len(self.trackers) > 0 and len(detections) > 0:
            row_idx, col_idx = linear_sum_assignment(cost)
            assigned_trks = set()
            assigned_dets = set()
            for i, j in zip(row_idx, col_idx):
                if cost[i, j] <= self.dist_threshold:  #  距离 <= 距离阈值  
                    matched.append((i, j))
                    assigned_trks.add(i)
                    assigned_dets.add(j)

            unmatched_trk = [i for i in range(len(self.trackers)) if i not in assigned_trks]
            unmatched_det = [j for j in range(len(detections)) if j not in assigned_dets]
        else:
            unmatched_det = list(range(len(detections)))
            unmatched_trk = list(range(len(self.trackers)))

        # Step 4: 更新已匹配
        for i, j in matched:
            self.trackers[i].update(detections[j],timestamp) 

        # Step 5: 新建未匹配的检测
        for j in unmatched_det:
            self.trackers.append(Tracker(np.array(detections[j]), self.next_id,timestamp=timestamp))
            self.next_id += 1

        # Step 6: 删除长期未更新的追踪器
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

    def pop_best_length_width_height(self, line):
        length = 999999999
        width = 999999999
        height = 999999999
        centre_l = 999999999
        centre_w = 999999999
        centre_h = 999999999
        best_speed  = 0
        best_idx = -1 

        for idx, trk in enumerate(self.trackers):
            if trk.popped:   # 跳过已标记的追踪器
                continue
            val = trk.get_last_observation()
            # [x,y,z,l,w,h,yaw,line]
            if val[0] < centre_l and val[7] == line:
                length   = trk.length
                width    = trk.width
                height   = trk.height
                centre_l = val[0]
                centre_w = val[1]
                centre_h = val[2]
                best_idx = idx   # 记录下标
                best_speed = trk.get_speed()

        if best_idx == -1:  # 没找到
            return 0, 0, 0, 0, 0, 0,best_speed
        print("get centre_l=",centre_l )
        self.trackers[best_idx].popped = True
        return length, width, height, centre_l, centre_w, centre_h,best_speed


# ==== Demo ====
if __name__ == "__main__":
    import time
    import random
    mot = MultiObjectTracker()

    t1 = int(time.time()*1000)
    detections_frame1 = [[-4.9793,7.0084,23.4296,4.4337,1.7374,1.6647,1.5855,0],[-4.3215,6.9752,38.2180,3.3681,1.6638,1.5606,1.6408,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 

    t1 = int(time.time()*1000)
    detections_frame1 = [[-4.9799,7.0066,23.4876,4.2820,1.7598,1.6414,1.5678,0],[-4.3315,6.9768,38.1654,3.5486,1.7053,1.5597,1.6191,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 
    
    t1 = int(time.time()*1000)
    detections_frame1 = [[-4.9409,7.0172,23.4819,4.5374,1.8164,1.6706,1.5526,0],[-4.3086,6.9892,38.3463,3.8089,1.7052,1.5637,1.6522,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 
    t1 = int(time.time()*1000)

    detections_frame1 = [[-4.9998,7.0204,23.3865,4.3963,1.7817,1.6059,1.5674,0],[-4.3539,6.9894,38.3511,3.6046,1.6689,1.5639,1.6327,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 
    
    t1 = int(time.time()*1000)
    detections_frame1 =[[-4.9871,7.0255,23.4278,4.3318,1.7970,1.6862,1.5622,0],[-4.2363,6.9774,38.3099,4.0176,1.6639,1.5271,1.6467,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 
    
    t1 = int(time.time()*1000)
    detections_frame1 =[[-4.9997,7.0289,23.3939,4.3464,1.8784,1.6850,1.5592,0],[-4.3242,6.9925,38.1914,3.7002,1.6784,1.5289,1.6380,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 
    
    t1 = int(time.time()*1000)
    detections_frame1 = [[-4.9620,7.0230,23.3777,4.2339,1.7142,1.6879,1.5631,0],[-4.3255,6.9896,38.2703,3.7798,1.7042,1.5523,1.6491,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 
    
    t1 = int(time.time()*1000)
    detections_frame1 = [[-4.9910,7.0073,23.4517,4.2428,1.8132,1.6598,1.5668,0],[-4.3490,6.9862,38.0772,3.7178,1.6914,1.5830,1.6356,0]]
    mot.update(detections_frame1, t1)
    print( "input: ",detections_frame1)
    time.sleep(random.uniform(0.2, 0.5)) 

    #print(mot.pop_best_length_width_height(0))
