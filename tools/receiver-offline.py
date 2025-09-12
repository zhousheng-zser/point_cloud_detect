import asyncio
import time
from collections import deque
from flask import Flask, request, jsonify
import threading
import numpy as np
import os
import datetime
from eval_rcnn import PointCloudConverter

def save_lidar_data_as_pcd(points):
    output_dir = '../csv_to_clouds'
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

def point_cloud_detect():

    PointCloud = PointCloudConverter()
    while True:
        try:
            import open3d as o3d  
            pcd = o3d.io.read_point_cloud("./test.pcd") 
            latest_points = np.asarray(pcd.points) 
            zeros_column = np.zeros((latest_points.shape[0], 1))
            latest_points = np.column_stack((latest_points, zeros_column))
            global status_unique_id
            results,result_lines = PointCloud.eval_one_epoch(latest_points)
            print(result_lines)
            save_lidar_data_as_pcd(results)
            time.sleep(1)

        except Exception as e:
            print(f"Error in point_cloud_detect processing loop: {e}")
            time.sleep(1) 

if __name__ == '__main__':
    
    detect_thread = threading.Thread(target=point_cloud_detect, daemon=True)
    detect_thread.start()

    detect_thread.join()