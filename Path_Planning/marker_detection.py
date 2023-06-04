#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated:
# this is the visual odometry node for Rosbots that processes image_raw_color messages

#### ROS node messages
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from geometry_msgs.msg import PoseStamped


#### other packages
import numpy as np
import cv2
import roslib
from cv_bridge import CvBridge, CvBridgeError
import sys
import os as os
import transformations as transf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from yaml.loader import SafeLoader
from multiprocessing import Process


DEFAULT_IMAGE_TOPIC = '/camera/rgb/image_rect_color'

class MarkerDetection:

    def __init__(self):
        # Setting up the publisher to send velocity commands.

        # Setting up subscriber receiving messages from the laser.
        self.cv_image = None
        self.image_message = None
        self.image_publisher = rospy.Publisher("output_image", Image)
        self.image_subscriber = rospy.Subscriber(DEFAULT_IMAGE_TOPIC, Image,
                                                 self.image_callback, queue_size=1)
        self.bridge = CvBridge()
        # Parameters.
        self.linear_velocity = 0  # Constant linear velocity set.
        self.angular_velocity = 0  # Constant angular velocity set.
        self.image_width = 0
        self.image_height = 0
        self.all_frames = []
        self.orb_feature_detector = cv2.ORB_create()
        self.previous_image = None
        self.previous_key_points = None
        self.previous_descriptors = None
        self.current_frame = None
        self.current_frame = None
        # list to store the relevant positions
        self.robot_position_list = []  # calculated robot position list from opencv
        self.ground_truth_list = []  # list of ground truths from the ros node

    def image_callback(self, image_message):
        self.image_height = image_message.height
        self.image_width = image_message.width
        self.image_message = image_message

    def spin(self):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(self.image_message, 'bgr8')
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image", self.cv_image)
        cv2.waitKey(3)

        try:
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.cv_image, 'bgr8'))
        except CvBridgeError as e:
            print(e)

def main(args):
    rospy.init_node('VisualOdometryNode', anonymous=True)
    vis_odom = MarkerDetection()
    print("visual odometry activated")
    rospy.sleep(2)

    try:
        print("spinning...")
        vis_odom.spin()

    except rospy.ROSInterruptException:
        rospy.logerr("Ros node interrupted")
        cv2.destroyAllWindows()


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass






