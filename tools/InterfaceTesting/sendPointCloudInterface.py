
import os
import json
import time
import threading
import requests
from datetime import datetime
from collections import OrderedDict
from typing import Dict,List
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
    def __init__(self, unique_id, length, width, height,centre_length,centre_width,centre_height,
                 coordinate_system="ECEF", sensor_type="LiDAR", bin_data=None):
        self.unique_id = unique_id
        self.length = length
        self.width = width
        self.height = height
        self.centre_length = centre_length
        self.centre_width = centre_width
        self.centre_height = centre_height
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
# pointcloud_data_dict: Dict[str, PointCloudDataStruct] = OrderedDict()
# data_conditions: Dict[str, threading.Condition] = {}

pointcloud_data_dict: List[Dict[str, PointCloudDataStruct]] = [OrderedDict() for _ in range(2)]
data_conditions: List[Dict[str, threading.Condition]] = [{} for _ in range(2)]
global_lock = threading.Lock()

def send_PointCloud_Data_Interface(point_cloud_data: PointCloudDataStruct, road_id):
    unique_id = point_cloud_data.unique_id
    print("point_cloud_data =" , point_cloud_data.length,point_cloud_data.width,point_cloud_data.height ,"unique_id =" , unique_id )
    with global_lock:
        if len(pointcloud_data_dict[road_id]) >= MAX_SIZE:
            pointcloud_data_dict[road_id].popitem(last=False)
        pointcloud_data_dict[road_id][unique_id] = point_cloud_data
        if unique_id not in data_conditions[road_id]:
            data_conditions[road_id][unique_id] = threading.Condition()
        condition = data_conditions[road_id][unique_id]
    with condition:
        condition.notify_all()
    #print(f"[INFO] pointcloud_data_dict {road_id}: {len(pointcloud_data_dict[road_id])}")

def async_forward_to_other_service(data: PointCloudDataStruct, road_id):
    def worker():
        web_url_list = config.get("web_url_list")
        if not web_url_list:
            print("[FORWARD] No 'web_url_list' in config.")
            return

        radar_points = [list(point) for point in data.bin_data]
        payload = {
            "vehicle_width": data.width,
            "vehicle_height": data.height,
            "vehicle_length": data.length,
            "vehicle_centre_width": data.centre_width,
            "vehicle_centre_height": data.centre_height,
            "vehicle_centre_length": data.centre_length,
            "vehicle_serial_number": data.unique_id,
            "vehicle_detect_time": data.timestamp,
            "vehicle_radar_points": radar_points
        }
        try:
            response = requests.post(web_url_list[road_id], json=payload, timeout=2)
            print(f"[FORWARD] Sent to {web_url_list[road_id]} - status: {response.status_code} text: {response.text} json: {response.json()}  ")
        except Exception as e:
            print(f"[FORWARD] Error sending to {web_url_list[road_id]} - {e}")

    threading.Thread(target=worker, daemon=True).start()

app = Flask(__name__)

@app.route('/pointcloud/detect', methods=['POST'])
def handle_point_cloud_request():
    real_ip = real_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    aii_url_list = config.get("aii_url_list")    
    if real_ip not in aii_url_list:
            print("[FORWARD] No",real_ip, "in sendHttpConfig.json:aii_url_list")
            return

    road_id = int(aii_url_list[real_ip])
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
    unique_ids_ = req_body.get("unique_id")
    time_str  = req_body.get("EventDetectTime") 
    dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    unique_ids = unique_ids_ + "_" + str(int(dt_obj.timestamp()))
    print("Client IP:", real_ip,"data:", data,"unique_ids:", unique_ids)

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
            if unique_id in pointcloud_data_dict[road_id]:
                obj = pointcloud_data_dict[road_id].pop(unique_id)
                responses[unique_id] = obj.to_dict()
                async_forward_to_other_service(obj,road_id)
                if unique_id in data_conditions[road_id]:
                    del data_conditions[road_id][unique_id]
                continue

            if unique_id not in data_conditions[road_id]:
                data_conditions[road_id][unique_id] = threading.Condition()
            condition = data_conditions[road_id][unique_id]

        with condition:
            remaining = timeout_seconds - (time.time() - start_time)
            if remaining > 0:
                condition.wait(timeout=remaining)

        with global_lock:
            if unique_id in pointcloud_data_dict[road_id]:
                obj = pointcloud_data_dict[road_id].pop(unique_id)
                responses[unique_id] = obj.to_dict()
                async_forward_to_other_service(obj,road_id)
                if unique_id in data_conditions[road_id]:
                    del data_conditions[road_id][unique_id]

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
