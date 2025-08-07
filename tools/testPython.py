import livox_sdk_wrapper_python as livox
import time
import threading

exit_flag = threading.Event()

def start_lidar():
    try:
        livox.init_livox_lidar_sdk_cplusplus_interface("hapconfig.json")
        print("Livox SDK started.")
    except Exception as e:
        print(f"Livox error: {e}")
        exit_flag.set()

def wait_for_exit():
    while not exit_flag.is_set():
        user_input = input("Type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            exit_flag.set()

def main():
    lidar_thread = threading.Thread(target=start_lidar)
    input_thread = threading.Thread(target=wait_for_exit)

    lidar_thread.start()
    input_thread.start()

    while not exit_flag.is_set():
        pc = livox.get_livox_lidar_pointcloud_data_interface()
        if pc:
            points = pc['points']
            reflectivity = pc['reflectivity']
            tag = pc['tag']
            timestamp = pc['timestamp']

            for i in range(min(5, points.shape[0])):
                x, y, z = points[i]
                r = reflectivity[i]
                t = tag[i]
                tt =timestamp[i]
                print(f"python-----------------[Main] Point {i}: x={x:.3f}, y={y:.3f}, z={z:.3f} | Reflectivity={r} | Tag={t} | timestamp={tt}")
        else:
            time.sleep(0.1)

    print("Shutting down Livox SDK...")
    livox.uninit_livox_lidar_sdk_cplusplus_interface()  
    lidar_thread.join()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()
