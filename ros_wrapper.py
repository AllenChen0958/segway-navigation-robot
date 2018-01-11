import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from collections import deque

class RosImageSubscriber(object):
    def __init__(self, user_callback, topic='/csi_cam/image_raw'):
        '''
        param user_callback: a function like f(cv_image)
        '''
        self.bridge = CvBridge()
        self.user_callback = user_callback
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
   
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.user_callback(cv_image)
        except CvBridgeError as e:
            print(e)

class RosRealsenseDepthImageSubscriber:

    def __init__(self, user_callback):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.callback)
        self.user_callback = user_callback

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            cv2.normalize(cv_image, cv_image, 0, 1, cv2.NORM_MINMAX)
            cv_image = 1.0 - cv_image
            cv_image *= 255.0
            cv_image = np.expand_dims(cv_image, axis=-1)
            self.user_callback(cv_image)
        except CvBridgeError as e:
            print(e)

class RosDepthImageSubscriber:

    def __init__(self, user_callback):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed/depth/depth_registered", Image, self.callback)
        self.user_callback = user_callback

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            min_val = 0
            max_val = 125
            min_val, max_val, min_loc, max_loc=cv2.minMaxLoc(cv_image)
            if min_val == max_val :
                min_val =0
                max_val=2
            min_val = float(min_val)
            max_val = float(max_val)
            tmp = cv_image
            tmp = tmp.astype(np.float32)
            #tmp = tmp * (255.0 / (max_val - min_val))
            tmp = cv2.convertScaleAbs(tmp, tmp, 255.0 / (max_val - min_val))
            tmp = tmp.astype(np.uint8)
            #cv2.convertTo(img_scaled_8u, CV_8UC1, 255. / (max_val - min_val))
            #cv_image = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
            #cv2.normalize(cv_image, cv_image, 0, 10000, cv2.NORM_MINMAX)
            tmp = np.expand_dims(tmp, axis=-1)
            self.user_callback(tmp)
        except CvBridgeError as e:
            print(e)


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
            tmp = cv2.convertScaleAbs(tmp, tmp, 255.0 / (max_val - min_val))
            tmp = tmp.astype(np.uint8)
            cv_image = np.expand_dims(tmp, axis=-1)                
            self.user_callback(cv_image)
        except CvBridgeError as e:
            print(e)

class RosImageQueue(RosImageSubscriber):
    def __init__(self, queue_size, frame_stack, user_callback, topic='/csi_cam/image_raw'):
        super(RosImageQueue, self).__init__(user_callback, topic)
        self.frame_stack = frame_stack
        self.queue_size = queue_size
        self.deque = deque(maxlen=self.queue_size)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.deque
            self.user_callback(cv_image)
        except CvBridgeError as e:
            print(e)

class RosRMPMotionPublisher(object):
    
    FORWARD = 0
    TURN_R = 1
    TURN_L = 2
    STOP_ACTION = 3

    def __init__(self, topic='/rmp220/base/vel_cmd', linear=0.4, angular=0.38):
        self.pub = rospy.Publisher(topic, TwistStamped, queue_size=1)
        self.linear = linear
        self.angular = angular

    def publish(self, cmd):
        vel = TwistStamped()
        if cmd == self.STOP_ACTION:
            vel.twist.linear.x  = 0.0
            vel.twist.angular.z = 0.0
        elif cmd == self.TURN_R:
            vel.twist.linear.x  = self.linear
            vel.twist.angular.z = self.angular
        elif cmd == self.TURN_L:
            vel.twist.linear.x  = self.linear
            vel.twist.angular.z = -self.angular
        else:
            vel.twist.linear.x  = self.linear
            vel.twist.angular.z = 0.0
        self.pub.publish(vel)

class RosKobukiMotionPublisher(object):
    
    FORWARD = 0
    TURN_R = 1
    TURN_L = 2
    STOP_ACTION = 3

    def __init__(self, topic='/mobile_base/commands/velocity', linear=0.3, angular=0.3):
        self.pub = rospy.Publisher(topic, Twist, queue_size=5)
        rospy.set_param('/mobile_base/cmd_vel_timeout', '0.7')
        self.linear = linear
        self.angular = angular

    def publish(self, cmd):
        vel = Twist()
        if cmd == self.STOP_ACTION:
            vel.linear.x  = 0.0
            vel.angular.z = 0.0
        elif cmd == self.TURN_R:
            vel.linear.x  = self.linear * 0.5
            vel.angular.z = -self.angular
        elif cmd == self.TURN_L:
            vel.linear.x  = self.linear * 0.5
            vel.angular.z = self.angular * 0.5
        else:
            vel.linear.x  = self.linear
            vel.angular.z = 0.0
        self.pub.publish(vel)

if __name__ == '__main__':
    import rospy
    p = RosRMPMotionPublisher()
    try:
        rospy.init_node('agent', anonymous=True)
        while True:
            p.publish(0)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
