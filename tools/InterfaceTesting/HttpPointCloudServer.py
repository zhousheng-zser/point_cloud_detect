from flask import Flask, request, jsonify
from datetime import datetime
from sendPointCloudInterface import PointCloudDataStruct, send_PointCloud_Data_Interface

app = Flask(__name__)

# 模拟一个存储或生成点云几何信息的函数
def get_point_cloud_info(unique_id):
    # 实际应用中可根据 unique_id 查询数据库或计算点云边界
    return {
        "unique_id": unique_id,
        "length": 5.0,
        "width": 2.5,
        "height": 3.0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "coordinate_system": "ECEF",
        "sensor_type": "LiDAR"
    }

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
    unique_id = req_body.get("unique_id")

    if not unique_id:
        return jsonify({
            "ret_type": "get_point_cloud_detect_response",
            "ret_header": {
                "code": 2,
                "message": "Missing unique_id"
            },
            "ret_body": {}
        }), 400

    # 构造响应体
    pointcloud_info = get_point_cloud_info(unique_id)

    response = {
        "ret_type": "get_point_cloud_detect_response",
        "ret_header": {
            "code": 0
        },
        "ret_body": {
            "PointCloudsMessage": pointcloud_info
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8100)
