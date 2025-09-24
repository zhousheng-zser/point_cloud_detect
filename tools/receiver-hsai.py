import asyncio
import time
import os
from collections import deque
import datetime
from flask import Flask, request, jsonify
import threading
import numpy as np
import traceback
from eval_rcnn import PointCloudConverter
from Tracking import MultiObjectTracker
import hsai_sdk_wrapper_python as hsai
from draw_meshlab import process_point_cloud_with_3d_boxes
from InterfaceTesting.sendPointCloudInterface import PointCloudDataStruct, send_PointCloud_Data_Interface,start_pointcloud_server

app1 = Flask("app1")
app2 = Flask("app2")

# Thread-safe queue with maximum length of 5
points_queue = deque(maxlen=5)
queue_lock = threading.Lock()
status_point_cloud = {}
status_unique_id = {}
# Under_processing: Processing in progress
# Processing_completed: Processing finished
# Processing_request: Processing requested

mot = MultiObjectTracker()

def start_lidar():
    try:
        hsai.init_hsai_lidar_sdk_cplusplus_interface()
        print("hsai SDK started.")
    except Exception as e:
        print(f"hsai error: {e}")

def save_lidar_data_as_pcd(points):
    output_dir = '../detect_clouds'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f'test_Lidar_{current_time}.pcd'
    
    filepath = os.path.join(output_dir, filename)
    
    # Prepare PCD header
    num_points = points.shape[0]
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.6f %.6f %.6f %.6f')
    
    print(f"Saved point cloud data to {filepath}")
    return filepath

async def column_stack(time_interval_ms = 100):
    pc = hsai.get_hsai_lidar_pointcloud_data_interface(time_interval_ms)
    while (not pc) or (len(pc['points']) < 100000) :
        time.sleep(time_interval_ms/1000.0)
        pc = hsai.get_hsai_lidar_pointcloud_data_interface(time_interval_ms)

    points_ = np.array(pc['points'])
    reflectivity_ = np.array(pc['reflectivity'])
    reflectivity_column = reflectivity_.reshape(-1, 1)
    points = np.column_stack((points_, reflectivity_column))
    points_list = []
    points_list.append(points)
    return np.vstack(points_list)

async def collect_points_task():
    time_interval_ms = 100
    while True:
        try:
            points = await column_stack(time_interval_ms=time_interval_ms)
            with queue_lock:
                points_queue.append(points)
                #save_lidar_data_as_pcd(points)
                #print(f"Added points to queue. Queue size: {len(points_queue)}")
            #await asyncio.sleep(1)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in collect_points_task: {e}")
        await asyncio.sleep(time_interval_ms/1000.0 *1.5)  

# def get_length_width_height(result_lines):
#     parts = result_lines.split()
#     length = float(parts[10])
#     width = float(parts[9])
#     height = float(parts[8]) 
#     centre_l = float(parts[13])
#     centre_w =float(parts[11])
#     centre_h = float(parts[12])

#     for i in range(len(parts)//16):
#         if float(parts[i*16+13])< centre_l:
#             length   = float(parts[i*16+10])
#             width    = float(parts[i*16+9])
#             height   = float(parts[i*16+8])
#             centre_l = float(parts[i*16+13])
#             centre_w = float(parts[i*16+11])
#             centre_h = float(parts[i*16+12])
#     return length,width,height,centre_l,centre_w,centre_h

def point_cloud_detect():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.ensure_future(collect_points_task())

    # Start event loop
    threading.Thread(target=loop.run_forever, daemon=True).start()

    PointCloud = PointCloudConverter()
    while True:
        try:
            with queue_lock:
                if points_queue:
                    latest_points = points_queue[-1]
                else:
                    time.sleep(0.1)
                    continue
            points,roads_roi, result_lines,road_list = PointCloud.eval_one_epoch(latest_points) 
            result_lines_= '\n'.join(result_lines)
            if result_lines_ :
                print(result_lines_)
                print("")
                detections_frame = []
                parts = result_lines_.split()
                for i in range(len(parts)//16):
                    detections_frame.append([float(parts[i*16+13]),float(parts[i*16+11]),float(parts[i*16+12]),
                                        float(parts[i*16+10]),float(parts[i*16+9]),float(parts[i*16+8]),float(parts[i*16+14]), road_list[i]  ] )
                mot.update(detections_frame)
            
            global status_point_cloud ,status_unique_id
            road_map={
                8991: 0,
                8992: 1 
                }
            for port, state in status_point_cloud.items():
                if state == 'Processing_request':#请求来了
                    # Update status to processing
                    status_point_cloud[port] = 'Under_processing' #处理
                    id_ = road_map[port]
                    result_lines_temp = [line for line, road in zip(result_lines, road_list) if road == id_]
                    line_roi = f"{'Car'} {-1} {-1} {0.0:.4f} {0.0:.4f} {0.0:.4f} {0.0:.4f} {0.0:.4f} {roads_roi[id_][0]:.4f} {roads_roi[id_][1]:.4f} {roads_roi[id_][2]:.4f} {roads_roi[id_][3]:.4f} {roads_roi[id_][4]:.4f} {roads_roi[id_][5]:.4f} {roads_roi[id_][6]:.4f} {10.0:.4f}"
                    result_lines_temp.append(line_roi)
                    results = process_point_cloud_with_3d_boxes(points, '\n'.join(result_lines_temp), calib_path='./cfgs/calib.txt')

                    length_,width_,height_, centre_length_,centre_width_,centre_height_ = mot.get_length_width_height(id_)
                    point_cloud_data = PointCloudDataStruct(unique_id = str(status_unique_id[port]),length=length_, width=width_,height=height_,
                                centre_length=centre_length_,centre_width=centre_width_,centre_height=centre_height_ ,bin_data = results )
                    send_PointCloud_Data_Interface(point_cloud_data,id_)
                    status_point_cloud[port] = 'Processing_completed'  ##结束

        except Exception as e:
            save_lidar_data_as_pcd(latest_points)
            traceback.print_exc()
            print(f"Error in point_cloud_detect processing loop: {e}")
            for port, state in status_point_cloud.items():
                status_point_cloud[port] = 'Processing_completed' 
            time.sleep(0.1) 

@app1.route('/s485', methods=['POST'])
def handle_s485_1():
    return handle_s485_common(8991)

@app2.route('/s485', methods=['POST'])
def handle_s485_2():
    return handle_s485_common(8992)
    
def handle_s485_common(port_name):
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": f"No JSON data received on {port_name}"}), 400
        if int(data["S485WeightMessage"]["VehicleStatus"]) != 1:
            return jsonify({"warning": f"VehicleStatus != 1 on {port_name}"}), 200  #对于大车容易只有半截

        print(f"[{port_name}] Received data: {data}")
        global status_point_cloud, status_unique_id
        while status_point_cloud[port_name] == 'Processing_request': #进行中就一直等
            time.sleep(0.01)
        
        status_point_cloud[port_name] = 'Processing_request'
        time_str  = data["S485WeightMessage"]["EventDetectTime"]
        dt_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        status_unique_id[port_name]= data["S485WeightMessage"]["UniqueId"]+"_"+str(int(dt_obj.timestamp()))
        print(f"[{port_name}] ID = {status_unique_id}")
            
        # Print received data (for debugging)
        return jsonify({"Response": "OK"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def start_pointcloud_detect(app, port_number):
    app.run(host='0.0.0.0', port=port_number)


if __name__ == '__main__':

    start_lidar()
    start_thread_v1 = threading.Thread(target=start_pointcloud_server, kwargs={"host": "0.0.0.0", "port": 8100}, daemon=True)

    
    status_point_cloud[8991]="Processing_completed"
    status_point_cloud[8992]="Processing_completed"
    status_unique_id[8991] = ""
    status_unique_id[8992] = ""
    start_thread_road1 = threading.Thread(target=start_pointcloud_detect,args=(app1, 8991), daemon=True)
    start_thread_road2 = threading.Thread(target=start_pointcloud_detect,args=(app2, 8992), daemon=True)

    start_thread_v1.start()
    start_thread_road1.start()
    start_thread_road2.start()
    
    time.sleep(5) 
    detect_thread = threading.Thread(target=point_cloud_detect, daemon=True)
    detect_thread.start()

    detect_thread.join()
    hsai.uninit_hsai_lidar_sdk_cplusplus_interface()