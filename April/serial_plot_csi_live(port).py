import os
import serial
import csv
import time

# Configuration parameters
custom_serial_port = '/dev/ttyUSB0'
custom_baud_rate = 1000000

# Serial port setup
ser = serial.Serial(custom_serial_port, custom_baud_rate)

# CSV file setup
csv_filename = f"csi_data_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'}.csv"
csv_path = os.path.join(os.getcwd(), csv_filename)

# Ensure file is open for the duration of the loop
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['CSI_DATA', 'PA', 'MAC', 'RSSI', 'Rate', 'Sig_Mode', 'MCS', 'Bandwidth', 'Smoothing',
                     'Not_Sounding', 'Aggregation', 'STBC', 'FEC_Coding', 'SGI', 'Noise_Floor', 'AMPDU_CNT',
                     'Channel', 'Secondary_Channel', 'Local_Timestamp', 'Ant', 'Sig_Len', 'RX_State',
                     'Real_Time_Set', 'Real_Timestamp', 'Len', 'CSI_Data', 'Timestamp'])

    try:
        while True:
            serial_line = ser.readline().decode().strip()
            if "CSI_DATA" in serial_line:
                all_data = serial_line.split(',')
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000}'
                timestamp = timestamp.replace(':', '-')
                timestamp = timestamp.replace('.', '-')
                writer.writerow(all_data + [timestamp])

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")
    except Exception as e:
        print("An error occurred:", e)
    finally:
        ser.close()
        print("CSV file saved at:", csv_path)
