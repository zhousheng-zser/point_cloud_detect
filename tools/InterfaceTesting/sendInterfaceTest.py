import threading
import time
from sendPointCloudInterface import PointCloudDataStruct, send_PointCloud_Data_Interface, start_pointcloud_server

def push_mock_data():
    for i in range(5000):  
        unique_id = str(1000 + i)
        print(f"[PUSH] Pushing data for unique_id: {unique_id}")
        point_cloud_data = PointCloudDataStruct(
            unique_id=unique_id,
            length=4.5 + i,
            width=2.0 + i * 0.1,
            height=2.8 + i * 0.2
        )
        send_PointCloud_Data_Interface(point_cloud_data)
        time.sleep(1)

def main():
    
    server_thread = threading.Thread(target=start_pointcloud_server, kwargs={"host": "0.0.0.0", "port": 8100}, daemon=True)
    server_thread.start()

    time.sleep(1)  
    threading.Thread(target =  push_mock_data).start()
    
    while True:
        time.sleep(60)  

if __name__ == "__main__":
    main()
