import cv2
import threading as th
import struct
import numpy as np
from datetime import datetime
import time

from tools.tasks.taskManager import taskManager
from tools.log.LogWriter import LogWriter
from tools.hqueue import hQueue


devices = [
    ['Dory1', '140.114.75.144', 8888],
]

result_queue = hQueue(capacity=30)

img_path = './input'


def task(img):
    _, img_bytes = cv2.imencode('.jpg', img)
    img_bytes = img_bytes.tostring()
    seg = taskManager.sendTask(img_bytes)
    seg = np.fromstring(seg, np.uint8)
    seg = cv2.imdecode(seg, 1)
    return seg

def callback(task, value, state):
    if state == 1:
        result_queue.push((task.create_time(), value))
    else:
        raise value


def get_from_queue():
    cur_time, seg = result_queue.pop()
    return seg

def main():
    tm = taskManager(num_threads=1, device_list=devices, log='device.log', name='tm')


    cap = cv2.VideoCapture('test_video.MOV')

    cur_time = datetime.now()

    start = time.time()

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        tm.addTask(task, callback, frame)

        if result_queue.isEmpty():
            time.sleep(0.033)
            continue
        
        #result_queue.discard_lt((cur_time, None))
        output = get_from_queue()
        print('fps = {0}'.format( 1/(time.time()-start) ))
        start = time.time()
        
        cv2.imshow('frame', output)
        cv2.waitKey(1)
        time.sleep(0.033)

        

    tm.waitCompletion()
    tm.closeDevice()
    

if __name__ == '__main__':
    main()
