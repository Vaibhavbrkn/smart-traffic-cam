from concurrent.futures import ThreadPoolExecutor
import time
from func import license_func


executors_list = []

with ThreadPoolExecutor(max_workers=5) as executor:
    executors_list.append(executor.submit(
        license_func, 'tf', './checkpoints/custom-416', 416, False, 'yolov4', './data/video/license_plate.mp4', './detections/recognition.avi', 'XVID', 0.45, 0.50, False, False, False, False, True))


for x in executors_list:
    print(x.result())
