import cv2
import threading as th
import struct
import numpy as np
from datetime import datetime
import time

from tools.tasks.taskManager import taskManager
from tools.log.LogWriter import LogWriter
from tools.hqueue import hQueue

import os
import os.path


devices = [
    ['Dory1', '140.114.75.144', 8888],
    ['Dory2', '140.114.75.144', 8889],
]

result_queue = hQueue(capacity=30)

img_path = './input'


def task(name, img):
    _, img_bytes = cv2.imencode('.jpg', img)
    img_bytes = img_bytes.tostring()
    seg = taskManager.sendTask(img_bytes)
    seg = np.fromstring(seg, np.uint8)
    seg = cv2.imdecode(seg, 1)
    return (name, seg)

def callback(task, value, state):
    if state == 1:
        print('push to result_queue')
        result_queue.push((task.create_time(), value))
    else:
        raise value


def get_from_queue():
    print('get seg')
    cur_time, out = result_queue.pop()
    name, seg = out
    return name, seg

def main():
    tm = taskManager(num_threads=3, device_list=devices, log='device.log', name='tm')


    files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    #cap = cv2.VideoCapture(0)

    cur_time = datetime.now()

    start = time.time()

    for i in range(len(files)):
        frame = cv2.imread(os.path.join(img_path, files[i]))

        print('add task')
        tm.addTask(task, callback, files[i], frame)

        if result_queue.isEmpty():
            continue
        
        #result_queue.discard_lt((cur_time, None))
        name, output = get_from_queue()
        
        cv2.imshow('frame', output)
        cv2.imwrite('./output/'+name, output)

    tm.waitCompletion()

    while not result_queue.isEmpty():
        name, output = get_from_queue()
        
        cv2.imshow('frame', output)
        cv2.imwrite('./output/'+name, output)


    print('spend {0} secs'.format(time.time()-start))
    tm.closeDevice()
    

if __name__ == '__main__':
    main()
