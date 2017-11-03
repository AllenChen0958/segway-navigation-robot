import rospy
import cv2
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
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

    def __init__(self, topic='/rmp220/base/vel_cmd', linear=0.2, angular=0.17):
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
