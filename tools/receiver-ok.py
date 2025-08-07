import asyncio
import time
from collections import deque
from flask import Flask, request, jsonify
import threading
import numpy as np
from eval_rcnn import PointCloudConverter
import livox_sdk_wrapper_python as livox
from InterfaceTesting.sendPointCloudInterface import PointCloudDataStruct, send_PointCloud_Data_Interface,start_pointcloud_server

app = Flask(__name__)

# Thread-safe queue with maximum length of 5
points_queue = deque(maxlen=5)
queue_lock = threading.Lock()
status_point_cloud = 'Processing_completed'
status_unique_id = ""
# Under_processing: Processing in progress
# Processing_completed: Processing finished
# Processing_request: Processing requested


def start_lidar():
    try:
        livox.init_livox_lidar_sdk_cplusplus_interface("hapconfig.json")
        print("Livox SDK started.")
    except Exception as e:
        print(f"Livox error: {e}")

async def column_stack(time_interval_ms = 100000000):
    pc = livox.get_livox_lidar_pointcloud_data_interface()
    begin_timestamp= -999999999999999999999
    if pc:
        begin_timestamp = pc['timestamp']
    points_list = []
    while True:
        if pc:
            timestamp = pc['timestamp']
            if(timestamp - begin_timestamp > time_interval_ms) :
                break
            
            points_ = np.array(pc['points'])
            reflectivity_ = np.array(pc['reflectivity'])
            reflectivity_column = reflectivity_.reshape(-1, 1)
            points = np.column_stack((points_, reflectivity_column))
            points_list.append(points)
        else:
            time.sleep(0.01)
        pc = livox.get_livox_lidar_pointcloud_data_interface()
    return np.vstack(points_list)

async def collect_points_task():
    time_interval_ms = 100*1000000
    while True:
        try:
            points = await column_stack(time_interval_ms=time_interval_ms)
            with queue_lock:
                points_queue.append(points)
                print(f"Added points to queue. Queue size: {len(points_queue)}")
            #await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in collect_points_task: {e}")
        await asyncio.sleep(1)  

def point_cloud_detect():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.ensure_future(collect_points_task())

    # Start event loop
    threading.Thread(target=loop.run_forever, daemon=True).start()

    PointCloud = PointCloudConverter()
    while True:
        try:
            global status_point_cloud
            if status_point_cloud == 'Processing_request':
                print("Processing request detected")
                # Update status to processing
                status_point_cloud = 'Under_processing'

                with queue_lock:
                    if points_queue:
                        latest_points = points_queue[-1]
                    else:
                        #import open3d as o3d   #######################
                        #pcd = o3d.io.read_point_cloud("./cloud_000000000000.pcd") #######################
                        #latest_points = np.asarray(pcd.points) #######################
                        #zeros_column = np.zeros((latest_points.shape[0], 1))
                        #latest_points = np.column_stack((latest_points, zeros_column))
                        #print("latest_points : " , latest_points)
                        print("Queue is empty, waiting for data")
                        status_point_cloud = 'Processing_completed'
                        #time.sleep(1)
                        continue
                global status_unique_id
                results,result_lines = PointCloud.eval_one_epoch(latest_points)
                point_cloud_data = PointCloudDataStruct(unique_id = str(status_unique_id),length=0.0, width=0.0, height=0.0, bin_data = results )
                print(result_lines)
                if result_lines  :
                    parts = result_lines.split()
                    point_cloud_data.length = float(parts[10]) 
                    point_cloud_data.width = float(parts[9])
                    point_cloud_data.height = float(parts[8])
                    print(f"{point_cloud_data.length}  {point_cloud_data.width} {point_cloud_data.height}")
                send_PointCloud_Data_Interface(point_cloud_data)
                status_point_cloud = 'Processing_completed'

            time.sleep(0.1)

        except Exception as e:
            print(f"Error in point_cloud_detect processing loop: {e}")
            status_point_cloud = 'Processing_completed' 
            time.sleep(0.1) 


@app.route('/s485', methods=['POST'])
def handle_s485():
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        print(f"Received data: {data}")
        global status_point_cloud
        global status_unique_id
        if status_point_cloud != 'Under_processing':
            status_point_cloud = 'Processing_request'
            status_unique_id= data["S485WeightMessage"]["UniqueId"]
        # Print received data (for debugging)
        return jsonify({"Response": "OK"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def start_pointcloud_detect():
    app.run(host='0.0.0.0', port=8991)


if __name__ == '__main__':

    start_lidar()
    start_thread_v1 = threading.Thread(target=start_pointcloud_server, kwargs={"host": "0.0.0.0", "port": 8100}, daemon=True)
    start_thread_v2 = threading.Thread(target=start_pointcloud_detect, daemon=True)

    start_thread_v1.start()
    start_thread_v2.start()
    
    time.sleep(5) 
    detect_thread = threading.Thread(target=point_cloud_detect, daemon=True)
    detect_thread.start()

    detect_thread.join()
    livox.uninit_livox_lidar_sdk_cplusplus_interface()