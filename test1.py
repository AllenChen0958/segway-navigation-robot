#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        #self.image_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, self.callback)
        self.image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")

            min_val = 0
            max_val = 10
            min_val, max_val, min_loc, max_loc=cv2.minMaxLoc(cv_image)
            if min_val == max_val :
                min_val = 0
                max_val = 2
            min_val = float(min_val)
            max_val = float(max_val)
            tmp = cv_image
            tmp = tmp.astype(np.float32)
            #tmp = tmp * (255.0 / (max_val - min_val))
            tmp = cv2.convertScaleAbs(tmp, tmp, 255.0 / (max_val - min_val))
            tmp = tmp.astype(np.uint8)
            #cv2.convertTo(img_scaled_8u, CV_8UC1, 255. / (max_val - min_val))
            cv_image = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
            print(cv_image.shape)
            #cv2.normalize(cv_image, cv_image, 0, 10000, cv2.NORM_MINMAX)
        except CvBridgeError as e:
            print(e)
        cv2.imwrite("depth_%d.png", cv_image)

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous = True)
    try:
        print('Depth viewe')
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
