sudo systemctl stop ModemManager.service
sudo systemctl disable ModemManager.service
sudo slcand -o -c -s8 -S 115200 ttyACM1 can0
sudo ip link set can0 type can bitrate 1000000
sudo ifconfig can0 up