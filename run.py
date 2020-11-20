from concurrent.futures import ThreadPoolExecutor
import time
from func import tracking_func


executors_list = []

with ThreadPoolExecutor(max_workers=5) as executor:
    executors_list.append(executor.submit(
        tracking_func, 'tf', './checkpoints/yolov4-416', 416, True, 'yolov4', './data/video/test.mp4', './outputs/tiny.avi', 'XVID', 0.45, 0.50, False, False, True, 'window1', 'cam1.txt', 'cam2.txt', 'out1.txt', './jumpes/1'))
    executors_list.append(executor.submit(tracking_func, 'tf', './checkpoints/yolov4-416', 416, True,
                                          'yolov4', './data/video/test.mp4', './outputs/tiny.avi', 'XVID', 0.45, 0.50, False, False, True, 'window2', 'cam3.txt', 'cam4.txt', 'out2.txt', './jumpes/2'))


for x in executors_list:
    print(x.result())
