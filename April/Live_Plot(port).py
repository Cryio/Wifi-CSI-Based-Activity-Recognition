import serial

port = '/dev/ttyUSB0' 
baud_rate = 1000000

ser = serial.Serial(port, baud_rate)

try:
    while True:
        line = ser.readline().decode().strip()
        
        print(line)
        
except KeyboardInterrupt:
    ser.close() 
