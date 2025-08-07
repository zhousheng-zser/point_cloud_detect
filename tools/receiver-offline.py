import asyncio
import time
from collections import deque
from flask import Flask, request, jsonify
import threading
import numpy as np
from eval_rcnn import PointCloudConverter

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

            time.sleep(1)

        except Exception as e:
            print(f"Error in point_cloud_detect processing loop: {e}")
            time.sleep(1) 

if __name__ == '__main__':
    
    detect_thread = threading.Thread(target=point_cloud_detect, daemon=True)
    detect_thread.start()

    detect_thread.join()