from __future__ import print_function

import os
import sys
import time
import scipy.io as sio
import cv2
import tensorflow as tf
import numpy as np
import cPickle as pickle

from tools.socket.serverSock import serverSock
from model import DeepLabResNetModel

os.environ['CUDA_VISIBLE_DEVICES']='1'

from robot_project import predictor


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 27
SAVE_DIR = './output/'
INPUT_DIR = './input/'
RESTORE_PATH = './restore_weights/'
matfn = 'color150_2.mat'

flags = tf.app.flags

flags.DEFINE_string('name', 'server', 'ID which will be used in log file')
flags.DEFINE_integer('port', 8888, 'Server socket port')
flags.DEFINE_string('logfile', './server.log', 'Log file')
flags.DEFINE_integer('width', 640, 'Width of input image')
flags.DEFINE_integer('height', 480, 'Height of input image')
flags.DEFINE_integer('device', 0, 'GPU device')
flags.DEFINE_string('path', './robot_project/train_log/train-unity3d-navigation_v3_rand_move_seg/model-64800.index', 'pretrained RL model path')
flags.DEFINE_boolean('rtimg', False, '')
flags.DEFINE_boolean('use_seg', True, '')

FLAGS = flags.FLAGS



def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    return color_table


label_colours = read_labelcolours(matfn)

def decode_labels(mask, num_classes=150):
    global label_colours

    n, h, w, c = mask.shape
    #assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    output = np.zeros((h, w, 3), dtype=np.uint8)
 
    for i in range(h):
        for j in range(w):
            k = mask[0, i, j, 0]
            output[i, j, 0] = label_colours[k, 2]
            output[i, j, 1] = label_colours[k, 1]
            output[i, j, 2] = label_colours[k, 0]

    return output



def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


    

def main(argv=None):
    print ('Use seg', FLAGS.use_seg)
    input_img = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, 3])


    # Create network.
    net = DeepLabResNetModel({'data': input_img}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc_out']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(input_img)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

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


    p = predictor.getPredictor(load_path=FLAGS.path, transform=False)


    # create server
    # TODO: Remove socket dependency
    # ---begin---
    server = serverSock(name=FLAGS.name)
    server.create(port=FLAGS.port)
    server.waitForClient()
    # ---end---

    img_counter = 0
    while True:
        print('wait for task')
	    # TODO: img from ros callback function
        img = server.recv()

        print('receive task')
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, 1)
        hi, wi, _ = img.shape
        store_img = img
        img_counter += 1
        img = cv2.resize(img, (FLAGS.width, FLAGS.height)).astype(float)
        raw_img_resize = img
        img = img - IMG_MEAN

        img = np.expand_dims(img, axis=0)
        #### segmentation
        preds = sess.run(pred, feed_dict={input_img: img})
        ####
        if FLAGS.use_seg:
            msk = decode_labels(preds, num_classes=NUM_CLASSES)
        else:
            msk = raw_img_resize
        msk = cv2.resize(msk, (84, 84))
	
        #z = np.zeros(shape=(84, 84, 4), dtype=np.float32)

	    #z[:, :, 0:3] = msk

        if FLAGS.rtimg == True:
            msk = cv2.resize(msk, (wi, hi))
            _, msk = cv2.imencode('.png', msk)
            msk = msk.tostring()

            # TODO: No use
            server.send(msk)
        else:
            msk = cv2.resize(msk, (84, 84))
            act = p(msk)
            cv2.imwrite('seg_image/raw_img_%05d_%d.png' % (img_counter, act), store_img)	
            print('predict: {}'.format(act))
            # if img_counter==5:
            #     time.sleep(5)
            act = pickle.dumps(act)

	        # TOOD: Send action by ROS
            server.send(act)

            print('task done')

    # TODO: No use
    server.close()

if __name__ == '__main__':
    tf.app.run()
