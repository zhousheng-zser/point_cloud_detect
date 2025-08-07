import socket
import struct
import zlib

SERVER_IP = '127.0.0.1'
SERVER_PORT = 9990

HEADER_FORMAT = '>HBBIIQ'  # 包头格式（大端）
FOOTER_FORMAT = '>IH'      # CRC32 + 包尾标识（大端）
HEADER_SIZE = 20
FOOTER_SIZE = 6
POINT_SIZE = 13

def parse_packet(packet):
    if len(packet) < HEADER_SIZE + FOOTER_SIZE:
        print("[CLIENT] 数据包太小，丢弃")
        return

    header = packet[:HEADER_SIZE]
    footer = packet[-FOOTER_SIZE:]
    data = packet[HEADER_SIZE:-FOOTER_SIZE]

    magic, version, dtype, data_len, packet_id, timestamp = struct.unpack(HEADER_FORMAT, header)
    crc_recv, tail_magic = struct.unpack(FOOTER_FORMAT, footer)

    # 检查标识和长度
    if magic != 0xAA55 or tail_magic != 0x55AA:
        print("[CLIENT] 包头或包尾标识不正确，丢弃")
        return

    if len(data) != data_len:
        print(f"[CLIENT] 数据长度不匹配 (预期 {data_len} 实际 {len(data)}), 丢弃")
        return

    if zlib.crc32(data) != crc_recv:
        print("[CLIENT] CRC 校验失败，丢弃")
        return

    print(f"\n[CLIENT] ✅ 接收到包 #{packet_id} | 时间戳: {timestamp} | 点数量: {data_len // POINT_SIZE}")

    for i in range(0, data_len, POINT_SIZE):
        x, y, z = struct.unpack('<fff', data[i:i+12])
        point_type = data[i+12]

def receive_loop(sock):
    buffer = bytearray()
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk

            while True:
                if len(buffer) < HEADER_SIZE:
                    break  # 等待更多数据

                _, _, _, data_len, _, _ = struct.unpack(HEADER_FORMAT, buffer[:HEADER_SIZE])
                total_packet_size = HEADER_SIZE + data_len + FOOTER_SIZE

                if len(buffer) < total_packet_size:
                    break  # 包未完整

                packet = buffer[:total_packet_size]
                parse_packet(packet)
                buffer = buffer[total_packet_size:]

        except Exception as e:
            print(f"[CLIENT] 异常断开: {e}")
            break

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print(f"[CLIENT] 正在连接服务器 {SERVER_IP}:{SERVER_PORT}...")
        sock.connect((SERVER_IP, SERVER_PORT))
        print("[CLIENT] 已连接，开始接收数据...")
        receive_loop(sock)
