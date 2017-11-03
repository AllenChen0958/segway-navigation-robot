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

from robot_project import predictor
from ros_wrapper import RosImageSubscriber, RosRMPMotionPublisher
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
FLAGS = flags.FLAGS


# Image quque
MAX_QUEUE_SIZE = 8
image_queue = deque(maxlen=MAX_QUEUE_SIZE)

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

    # Create network.
    net = DeepLabResNetModel({'data': input_img}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc_out']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(input_img)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)

    # Color transform
    color_mat = label_colours[..., [2,1,0]]
    color_mat = tf.constant(color_mat, dtype=tf.float32)
    onehot_output = tf.one_hot(raw_output_up, depth=len(label_colours))
    onehot_output = tf.reshape(onehot_output, (-1, len(label_colours)))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, FLAGS.height, FLAGS.width, 3))
 
    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    ckpt = tf.train.get_checkpoint_state(RESTORE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    #p = predictor.getSingleFramePredictor(load_path=FLAGS.path, transform=False)
    p = predictor.getPredictor(load_path=FLAGS.path, transform=False)
    scale = 1.0
    ros_rmp_motion_publisher = RosRMPMotionPublisher(linear=0.2*scale, angular=0.15*scale)
    
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
            raw_img = cv2.resize(img, (FLAGS.width, FLAGS.height)).astype(float)
            img = raw_img - IMG_MEAN
            img = np.expand_dims(img, axis=0)
            preds = sess.run(pred, feed_dict={input_img: img})    
            if FLAGS.use_seg:
                s = preds[0].astype(np.uint8)
            else:
                s = raw_img
            msk = cv2.resize(s, (84, 84))
            act = p(msk)
            ros_rmp_motion_publisher.publish(act)
            end = time.time()
            print('Inference time = %f' % (end - start))
            
            if timestep < 1000:
                cv2.imwrite('raw_img_%05d_%d.png' % (timestep, act), raw_img)
                cv2.imwrite('msk_img_%05d_%d.png' % (timestep, act), s)
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
