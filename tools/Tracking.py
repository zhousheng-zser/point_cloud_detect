import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ==== 单个目标跟踪器 ====
class Tracker:
    def __init__(self, bbox3d, id):
        """
        bbox3d: [x, y, z, l, w, h, yaw,line]
        """
        self.last_observation = np.array(bbox3d)
        self.id = id
        self.time_since_update = 0
        self.hits = 1

        # ==== Kalman Filter ====
        # 状态: [x, y, z, vx, vy, vz, l, w, h, yaw, vyaw, line]
        # 11维
        self.kf = KalmanFilter(dim_x=12, dim_z=8)
        
        # 初始状态
        self.kf.x[:3] = bbox3d[:3].reshape(3, 1)   # 位置
        self.kf.x[6:9] = bbox3d[3:6].reshape(3, 1) # 尺寸
        self.kf.x[9] = bbox3d[6]                   # yaw
        self.kf.x[11] = bbox3d[7]                  # line
        
        # 状态转移矩阵 F
        self.kf.F = np.eye(12)
        dt = 1.0
        for i in range(3):
            self.kf.F[i, i+3] = dt  # 位置由速度更新
        self.kf.F[9, 10] = dt       # yaw 由角速度更新

        # 观测矩阵 H
        self.kf.H = np.zeros((8, 12))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 6] = 1  # l
        self.kf.H[4, 7] = 1  # w
        self.kf.H[5, 8] = 1  # h
        self.kf.H[6, 9] = 1  # yaw
        self.kf.H[7, 11] = 1  # line

        # 噪声
        self.kf.P *= 10.   # 初始不确定性
        self.kf.R *= 1.0   # 观测噪声
        self.kf.Q *= 0.01  # 过程噪声

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox3d):
        bbox3d = np.array(bbox3d).copy()
        bbox3d[3] = max(bbox3d[3], self.last_observation[3])  # l
        bbox3d[4] = max(bbox3d[4], self.last_observation[4])  # w
        bbox3d[5] = max(bbox3d[5], self.last_observation[5])  # h

        self.kf.update(bbox3d.reshape(-1, 1))
        self.time_since_update = 0
        self.hits += 1
        self.last_observation = bbox3d

    def get_state(self):
        """
        返回预测的 [x,y,z,l,w,h,yaw,line]
        """
        state = self.kf.x.flatten()
        return np.array([state[0], state[1], state[2],
                         state[6], state[7], state[8],
                         state[9], state[11]])
    def get_last_observation(self):
        """
        返回最近一次 update() 的观测值 [x,y,z,l,w,h,yaw,line]
        """
        return self.last_observation

# ==== 多目标跟踪管理器 ====
class MultiObjectTracker:
    def __init__(self, dist_threshold=1.5, max_age=2,iou_threshold=0.20):
        self.trackers = []
        self.next_id = 0
        self.dist_threshold = dist_threshold
        self.iou_threshold = iou_threshold
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
        """两凸多边形相交 (Sutherland–Hodgman)"""
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
        if b1[7]!= b1[7]:
            return 99999999
        dist = np.linalg.norm(np.array(b1[:3]) - np.array(b2[:3]))
        return dist

    def update(self, detections):
        """
        detections: [[x,y,z,l,w,h,yaw,line], ...]
        """
        # Step 1: 预测
        predicted_states = [trk.predict() for trk in self.trackers]

        # Step 2: 构建代价矩阵 (1 - IoU)
        cost = np.zeros((len(self.trackers), len(detections)))
        for i, trk in enumerate(self.trackers):
            for j, det in enumerate(detections):
                dist = self.get_iou(*trk.get_state(), *det)
                cost[i, j] = dist

        # Step 3: 匹配
        matched, unmatched_trk, unmatched_det = [], [], []
        if len(self.trackers) > 0 and len(detections) > 0:
            row_idx, col_idx = linear_sum_assignment(cost)

            for i, j in zip(row_idx, col_idx):
                if cost[i, j] >= self.iou_threshold:  # IoU >= 阈值
                    matched.append((i, j))
                else:
                    unmatched_trk.append(i)
                    unmatched_det.append(j)

            for i in range(len(self.trackers)):
                if i not in row_idx:
                    unmatched_trk.append(i)
            for j in range(len(detections)):
                if j not in col_idx:
                    unmatched_det.append(j)
        else:
            unmatched_det = list(range(len(detections)))
            unmatched_trk = list(range(len(self.trackers)))

        # Step 4: 更新已匹配
        for i, j in matched:
            self.trackers[i].update(detections[j])  #######################长宽高取max 

        # Step 5: 新建未匹配的检测
        for j in unmatched_det:
            self.trackers.append(Tracker(np.array(detections[j]), self.next_id))
            self.next_id += 1

        # Step 6: 删除长期未更新的追踪器
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

        # 输出所有活跃目标
        results = []
        for trk in self.trackers:
            results.append((trk.id, trk.get_state()))
        return results

    def pop_best_length_width_height(self, line):
        length = 999999999
        width = 999999999
        height = 999999999
        centre_l = 999999999
        centre_w = 999999999
        centre_h = 999999999
        best_idx = -1 

        for idx, trk in enumerate(self.trackers):
            val = trk.get_last_observation()
            # [x,y,z,l,w,h,yaw,line]
            if val[0] < centre_l and val[7] == line:
                length   = val[3]
                width    = val[4]
                height   = val[5]
                centre_l = val[0]
                centre_w = val[1]
                centre_h = val[2]
                best_idx = idx   # 记录下标

        if best_idx == -1:  # 没找到
            return 0, 0, 0, 0, 0, 0

        del self.trackers[best_idx]
        return length, width, height, centre_l, centre_w, centre_h


# ==== Demo ====
if __name__ == "__main__":
    mot = MultiObjectTracker()

    # 假设每一帧有检测框 (x,y,z,l,w,h,yaw)
    detections_frame1 = [[-5.0701, 7.0206, 34.2912,  10.2880,  2.9002, 4.0165,   1.6004, 0 ]]
    results = mot.update(detections_frame1)
    print("Frame 1:", results)

    detections_frame2 = [[-5.1582,7.0135, 32.4174, 10.3303, 2.8817, 4.0313, 1.6046,0 ]]
    results = mot.update(detections_frame2)
    print("Frame 2:", results)
