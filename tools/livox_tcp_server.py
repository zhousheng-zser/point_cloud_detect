import socket
import struct
import time
import csv
import zlib
import threading
from collections import defaultdict
import livox_sdk_wrapper_python as livox

HOST = '0.0.0.0'
PORT = 9990
CSV_FILE = '2025-04-22_18-45-07Test.Csv'
FRAME_SIZE = 91000
MODE = 'realtime'  # 'file' or 'realtime'

HEADER_MAGIC = 0xAA55
FOOTER_MAGIC = 0x55AA
VERSION = 0x01
DATA_TYPE = 0x01

exit_flag = threading.Event()

def build_packet(packet_id, timestamp_ms, points):
    header = struct.pack('>HBBIIQ', HEADER_MAGIC, VERSION, DATA_TYPE,
                         len(points) * 13, packet_id, timestamp_ms)

    body = bytearray()
    for x, y, z, type_byte in points:
        body += struct.pack('<fffB', x, y, z, type_byte)

    crc32 = zlib.crc32(body)
    footer = struct.pack('>I', crc32) + struct.pack('>H', FOOTER_MAGIC)
    return header + body + footer

def read_xyz_from_csv(csv_reader):
    frame = []
    for row in csv_reader:
        try:
            x = float(row[8])
            y = float(row[9])
            z = float(row[10])
            frame.append((x, y, z, 0x01))
            if len(frame) == FRAME_SIZE:
                yield frame
                frame = []
        except (IndexError, ValueError):
            continue
    if frame:
        yield frame

# ===== 实时模式使用的缓冲和打包逻辑 =====
frame_buffer = defaultdict(list)
lock = threading.Lock()

def livox_worker():
    livox.init_livox_lidar_sdk_cplusplus_interface("hapconfig.json")
    print("[LIVOX] SDK started.")

    while not exit_flag.is_set():
        pc = livox.get_livox_lidar_pointcloud_data_interface()
        if not pc:
            time.sleep(0.01)
            continue

        points = pc['points']
        tag = pc['tag']
        timestamp_ns = pc['timestamp']
        bucket = timestamp_ns // 300_000_000

        with lock:
            for i in range(points.shape[0]):
                x, y, z = points[i]
                t = tag[i]
                frame_buffer[bucket].append((x, y, z, t))
        
            # print(f"[DEBUG] bucket={bucket}, added={points.shape[0]}, total_in_bucket={len(frame_buffer[bucket])}")
            # 清理旧 bucket，只保留最新的 2 个 bucket（可调）
            if len(frame_buffer) > 200:
                max_b = max(frame_buffer.keys())
                for b in list(frame_buffer.keys()):
                    if b < max_b - 1:
                        del frame_buffer[b]

    livox.uninit_livox_lidar_sdk_cplusplus_interface()
    print("[LIVOX] SDK shutdown.")

def handle_client(conn, addr):
    print(f"[SERVER] Connected by {addr}")
    with conn:
        try:
            packet_id = 1
            if MODE == 'file':
                while True:
                    with open(CSV_FILE, newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        next(reader, None)
                        for frame in read_xyz_from_csv(reader):
                            timestamp_ms = int(time.time() * 1000)
                            packet = build_packet(packet_id, timestamp_ms, frame)
                            conn.sendall(packet)
                            print(f"[FILE] Sent packet #{packet_id}")
                            packet_id += 1
                            time.sleep(0.1)
            else:  # realtime 模式：只发最新一帧
                while not exit_flag.is_set():
                    to_send = None
                    with lock:
                        if frame_buffer:
                            latest_bucket = max(frame_buffer.keys())
                            frame = frame_buffer.pop(latest_bucket)
                            timestamp_ms = latest_bucket * 300
                            to_send = (timestamp_ms, frame)

                    if to_send:
                        timestamp_ms, frame = to_send
                        # print(f"[REALTIME] Packet #{packet_id} Preview (First 5 points):")
                        # for p in frame[:5]:
                            # print(f"  x={p[0]:.3f}, y={p[1]:.3f}, z={p[2]:.3f}, type={p[3]}")
                        packet = build_packet(packet_id, timestamp_ms, frame)
                        conn.sendall(packet)
                        print(f"[REALTIME] Sent packet #{packet_id}, points={len(frame)}")
                        packet_id += 1

                    time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[SERVER] Client {addr} disconnected.")
        except Exception as e:
            print(f"[SERVER] Error with client {addr}: {e}")

def start_server():
    if MODE == 'realtime':
        threading.Thread(target=livox_worker, daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        print(f"[SERVER] Listening on {HOST}:{PORT} in mode: {MODE}...")

        while True:
            conn, addr = server_sock.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == '__main__':
    start_server()
