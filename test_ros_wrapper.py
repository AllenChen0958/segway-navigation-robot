import rospy
from ros_wrapper import *
 
count = 0
if __name__ == '__main__':
    rospy.init_node('agent', anonymous=True)
    pub = RosRMPMotionPublisher(linear=0.1, angular=0.0)
  
    def _sub_callback(img):
        global count
        if count > 10:
            pub.publish(3)
        else:
            pub.publish(0)
        count += 1
        rospy.loginfo('Published: %d' % count)

    sub = RosImageSubscriber(user_callback=_sub_callback)
    rospy.spin()
