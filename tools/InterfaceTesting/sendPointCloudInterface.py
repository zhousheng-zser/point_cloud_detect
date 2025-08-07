
import os
import json
import time
import threading
import requests
from datetime import datetime
from collections import OrderedDict
from typing import Dict
from flask import Flask, request, jsonify

config = {}

def load_config():
    global config
    config_path = os.path.join(os.path.dirname(__file__), "sendHttpConfig.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[CONFIG] Loaded config from {config_path}")
    except Exception as e:
        print(f"[CONFIG] Failed to load config: {e}")
        config = {}

class PointCloudDataStruct:
    def __init__(self, unique_id, length, width, height,
                 coordinate_system="ECEF", sensor_type="LiDAR", bin_data=None):
        self.unique_id = unique_id
        self.length = length
        self.width = width
        self.height = height
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.coordinate_system = coordinate_system
        self.sensor_type = sensor_type
        self.bin_data = bin_data if bin_data is not None else []  # [(x, y, z, type)]

    def to_dict(self):
        return {
            "unique_id": self.unique_id,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "timestamp": self.timestamp,
            "coordinate_system": self.coordinate_system,
            "sensor_type": self.sensor_type
        }

MAX_SIZE = 30
pointcloud_data_dict: Dict[str, PointCloudDataStruct] = OrderedDict()
data_conditions: Dict[str, threading.Condition] = {}
global_lock = threading.Lock()

def send_PointCloud_Data_Interface(point_cloud_data: PointCloudDataStruct):
    unique_id = point_cloud_data.unique_id
    with global_lock:
        if len(pointcloud_data_dict) >= MAX_SIZE:
            pointcloud_data_dict.popitem(last=False)
        pointcloud_data_dict[unique_id] = point_cloud_data
        if unique_id not in data_conditions:
            data_conditions[unique_id] = threading.Condition()
        condition = data_conditions[unique_id]
    with condition:
        condition.notify_all()
    print(f"[INFO] pointcloud_data_dict: {len(pointcloud_data_dict)}")

def async_forward_to_other_service(data: PointCloudDataStruct):
    def worker():
        forward_url = config.get("forward_url")
        if not forward_url:
            print("[FORWARD] No 'forward_url' in config.")
            return

        radar_points = [list(point) for point in data.bin_data]

        payload = {
            "vehicle_width": int(data.width),
            "vehicle_height": int(data.height),
            "vehicle_length": int(data.length),
            "vehicle_serial_number": int(data.unique_id),
            "vehicle_detect_time": data.timestamp,
            "vehicle_radar_points": radar_points
        }

        try:
            response = requests.post(forward_url, json=payload, timeout=2)
            print(f"[FORWARD] Sent to {forward_url} - status: {response.status_code}")
        except Exception as e:
            print(f"[FORWARD] Error sending to {forward_url} - {e}")

    threading.Thread(target=worker, daemon=True).start()

app = Flask(__name__)

@app.route('/pointcloud/detect', methods=['POST'])
def handle_point_cloud_request():
    data = request.get_json()

    if not data or data.get("req_type") != "get_point_cloud_detect_request":
        return jsonify({
            "ret_type": "get_point_cloud_detect_response",
            "ret_header": {
                "code": 1,
                "message": "Invalid request"
            },
            "ret_body": {}
        }), 400

    req_body = data.get("req_body", {})
    unique_ids = req_body.get("unique_id")
    if isinstance(unique_ids, str):
        unique_ids = [unique_ids]

    if not unique_ids:
        return jsonify({
            "ret_type": "get_point_cloud_detect_response",
            "ret_header": {
                "code": 2,
                "message": "Missing unique_id"
            },
            "ret_body": {}
        }), 400

    timeout_seconds = 5
    start_time = time.time()
    responses = {}

    for unique_id in unique_ids:
        with global_lock:
            if unique_id in pointcloud_data_dict:
                obj = pointcloud_data_dict.pop(unique_id)
                responses[unique_id] = obj.to_dict()
                async_forward_to_other_service(obj)
                continue

            if unique_id not in data_conditions:
                data_conditions[unique_id] = threading.Condition()
            condition = data_conditions[unique_id]

        with condition:
            remaining = timeout_seconds - (time.time() - start_time)
            if remaining > 0:
                condition.wait(timeout=remaining)

        with global_lock:
            if unique_id in pointcloud_data_dict:
                obj = pointcloud_data_dict.pop(unique_id)
                responses[unique_id] = obj.to_dict()
                async_forward_to_other_service(obj)

    if not responses:
        return jsonify({
            "ret_type": "get_point_cloud_detect_response",
            "ret_header": {
                "code": 3,
                "message": f"Timeout waiting for unique_id(s): {unique_ids}"
            },
            "ret_body": {}
        }), 504

    return jsonify({
        "ret_type": "get_point_cloud_detect_response",
        "ret_header": {
            "code": 0
        },
        "ret_body": {
            "PointCloudsMessage": responses[unique_id]
        }
    }), 200

def start_pointcloud_server(host='0.0.0.0', port=8100):
    load_config()
    print(f"[INFO] HTTP: http://{host}:{port} Server started!")
    app.run(host=host, port=port)

if __name__ == '__main__':
    load_config()
    start_pointcloud_server()
