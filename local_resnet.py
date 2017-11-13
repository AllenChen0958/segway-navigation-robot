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

from model import DeepLabResNetModel

os.environ['CUDA_VISIBLE_DEVICES']='0'

from robot_project import predictor_resnet as predictor
from ros_wrapper import RosImageSubscriber, RosKobukiMotionPublisher, RosRMPMotionPublisher
import rospy

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 27
SAVE_DIR = './output/'
INPUT_DIR = './input/'
RESTORE_PATH = './restore_weights/'
matfn = 'color150_2.mat'

flags = tf.app.flags
flags.DEFINE_string('name', 'server', 'ID which will be used in log file')
flags.DEFINE_string('logfile', './server.log', 'Log file')
flags.DEFINE_integer('width', 640, 'Width of input image')
flags.DEFINE_integer('height', 480, 'Height of input image')
flags.DEFINE_integer('device', 0, 'GPU device')
flags.DEFINE_string('path', '', 'pretrained RL model path')
flags.DEFINE_boolean('perf', False, '')
flags.DEFINE_boolean('use_seg', True, '')
flags.DEFINE_integer('qsize', 8, 'Queue size')
flags.DEFINE_float('lin', 0.4, 'Linear')
flags.DEFINE_float('ang', 0.35, 'Angular')
FLAGS = flags.FLAGS

# Image quque
MAX_QUEUE_SIZE = FLAGS.qsize
image_queue = deque(maxlen=MAX_QUEUE_SIZE)
print('Image queue size = %d' % MAX_QUEUE_SIZE)

def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    return color_table

label_colours = read_labelcolours(matfn)

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

    p = predictor.getPredictor(load_path=FLAGS.path, transform=False)
    scale = 1.0
    #ros_rmp_motion_publisher = RosKobukiMotionPublisher()
    ros_rmp_motion_publisher = RosKobukiMotionPublisher(linear=FLAGS.lin, angular=FLAGS.ang)
    #ros_rmp_motion_publisher = RosRMPMotionPublisher(linear=FLAGS.lin, angular=FLAGS.ang)
    
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
            
            img = cv2.resize(img, (224, 224))
            s = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            act = p(s)
            ros_rmp_motion_publisher.publish(act)
            end = time.time()
            print('Inference time = %f' % (end - start))
            
            if timestep < 5:
                cv2.imwrite('raw_img_%05d_%d.png' % (timestep, act), raw_img)
                cv2.imwrite('msk_img_%05d_%d.png' % (timestep, act), s)
            timestep += 1
            print("STEP: {}".format(timestep))
            

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
