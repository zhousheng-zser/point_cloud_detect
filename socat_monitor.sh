#!/bin/bash 
#Forward TCP and UDP data to docker 

#CMD1="socat -d -d UDP4-RECV:2368,reuseaddr UDP4-SENDTO:172.17.0.2:2368"
CMD1="socat -u UDP4-RECV:2368,reuseaddr,rcvbuf=1048576 UDP4-SENDTO:172.17.0.2:2368"
CMD2="socat -d -d TCP-LISTEN:8991,reuseaddr,keepalive,fork TCP:172.17.0.2:8991"
CMD3="socat -d -d TCP-LISTEN:80,reuseaddr,keepalive,fork TCP:172.17.0.2:8100"

start_cmd1() {
    $CMD1 &
    PID1=$!
    echo "Started CMD1, PID=$PID1"
}
start_cmd2() {
    $CMD2 &
    PID2=$!
    echo "Started CMD2, PID=$PID2"
}
start_cmd3() {
    $CMD3 &
    PID3=$!
    echo "Started CMD3, PID=$PID3"
}

start_cmd1
start_cmd2
start_cmd3

while true; do
    if ! kill -0 $PID1 2>/dev/null; then
        echo "CMD1 stopped, restarting..."
        start_cmd1
    fi
    if ! kill -0 $PID2 2>/dev/null; then
        echo "CMD2 stopped, restarting..."
        start_cmd2
    fi
    if ! kill -0 $PID3 2>/dev/null; then
        echo "CMD3 stopped, restarting..."
        start_cmd3
    fi
    sleep 5
done
