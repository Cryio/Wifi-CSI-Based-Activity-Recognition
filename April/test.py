from datetime import datetime 
import time

print(datetime.strptime(str(datetime.now()) , '%Y-%m-%d %H:%M:%S.%f')) 
# print(str(time.time()))