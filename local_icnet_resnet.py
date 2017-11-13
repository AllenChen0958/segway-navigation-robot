from __future__ import print_function

import os
import sys
import time
import scipy.io as sio
import cv2
import tensorflow as tf
import numpy as np
import cPickle as pickle

from collections import deque
import threading

from model_icnet import ICNet

os.environ['CUDA_VISIBLE_DEVICES']='0'

from robot_project import predictor_resnet as predictor
from ros_wrapper import RosImageSubscriber, RosKobukiMotionPublisher, RosRMPMotionPublisher
import rospy

NUM_CLASSES = 19
IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

SAVE_DIR = './output/'
INPUT_DIR = './input/'

RESTORE_PATH = './restore_weights/icnet_cityscapes_trainval_90k.npy'


flags = tf.app.flags
flags.DEFINE_string('name', 'server', 'ID which will be used in log file')
flags.DEFINE_string('logfile', './server.log', 'Log file')
flags.DEFINE_integer('width', 640, 'Width of input image')
flags.DEFINE_integer('height', 480, 'Height of input image')
flags.DEFINE_integer('device', 0, 'GPU device')
flags.DEFINE_string('path', '', 'pretrained RL model path')
flags.DEFINE_boolean('perf', False, '')
flags.DEFINE_boolean('use_seg', True, '')
FLAGS = flags.FLAGS


# Image quque
MAX_QUEUE_SIZE = 2
image_queue = deque(maxlen=MAX_QUEUE_SIZE)

'''
label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32), (1, 1, 1)]
                # 18 = bicycle, 19 = void label
'''
label_colours = [(0, 255, 0), (0, 0, 255), (255, 0, 0)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(255, 0, 0), (255, 0, 0), (255, 0, 0)
                # 3 = wall, 4 = fence, 5 = pole
                ,(0, 0, 255), (0, 0, 255), (0, 0, 255)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(255, 0, 0), (255, 0, 255), (255, 0, 0)
                # 9 = terrain, 10 = sky, 11 = person
                ,(0, 0, 255), (0, 0, 255), (0, 0, 255)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 0, 255), (0, 0, 255), (0, 0, 255)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(0, 0, 255), (255, 0, 255)]
                # 18 = bicycle, 19 = void label

label_colours = np.array(label_colours)

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main(argv=None):
    # Input placeholder
    input_img = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, 3])
 
    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    #p = predictor.getSingleFramePredictor(load_path=FLAGS.path, transform=False)
    p = predictor.getPredictor(load_path=FLAGS.path, transform=False)
    scale = 1.0
    #ros_rmp_motion_publisher = RosKobukiMotionPublisher()
    ros_rmp_motion_publisher = RosRMPMotionPublisher()
    
    def actor(stop_event):
        global image_queue
        while len(image_queue) == 0:
            print('Wait for image queue filled ... (sleep 1)')
            time.sleep(1)
            pass
        timestep = 0 
        while not stop_event.is_set():
            start = time.time()
            img = image_queue[0]
            raw_img = img
            #raw_img = cv2.resize(img, (FLAGS.width, FLAGS.height)).astype(float)
            img = np.expand_dims(img, axis=0)
            s = raw_img
            msk_bgr = cv2.resize(s, (224, 224))
            # Change BGR -> RGB
            msk = cv2.cvtColor(msk_bgr, cv2.COLOR_BGR2RGB)
            act = p(msk)
            ros_rmp_motion_publisher.publish(act)
            end = time.time()
            print('Timestep = %d: Inference time = %f' % (timestep, end - start))
            
            #if timestep < 1000:
            #    cv2.imwrite('raw_img_%05d_%d.png' % (timestep, act), raw_img)
            #    cv2.imwrite('msk_img_%05d_%d.png' % (timestep, act), msk_bgr)
            timestep += 1
            

    def image_subscribe_callback(img):
        global image_queue
        image_queue.append(img)

    ros_camera_image_subscriber = RosImageSubscriber(user_callback=image_subscribe_callback)
    rospy.init_node('agent', anonymous=True)

    actor_thread_stop_event = threading.Event()
    try:
        actor_thread = threading.Thread(target=actor, args=(actor_thread_stop_event,))
        actor_thread.start()

        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        actor_thread_stop_event.set()
        actor_thread.join()

if __name__ == '__main__':
    tf.app.run()
