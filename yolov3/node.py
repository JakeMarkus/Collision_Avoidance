import rospy
from PIL import Image
from sensor_msgs.msg import Image as ros_image
from std_msgs.msg import Float64MultiArray
import numpy as np
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

counter = 0
class ROSBotDetectionNode:
    def __init__(self):
        print("init")
        rospy.init_node("collision_detection_bot")
        rospy.Subscriber("/camera/rgb/image_rect_color", ros_image, self.image_callback, queue_size=1)
        self.msg_pub = rospy.Publisher("rosbot_detection", Float64MultiArray, queue_size=10)
        rospy.spin()
        
    def compress(self, img_msg, count):
        global counter
        print("compress")     
    
        try:
            cv_image = CvBridge().imgmsg_to_cv2(img_msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
    
        color_converted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_converted)
     
        image.save(str(counter) + '.jpg')


    def image_callback(self, msg):
        global counter
        print("image_callback")
        self.compress(msg,counter)
        
        # command
        cmd = "python3 detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --save-txt --source " + str(counter) + ".jpg"
        os.system(cmd)
        
        final = []
        
        # Open the file for reading
        with open('runs/detect/exp/labels/' + str(counter) + '.txt', 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into a list of numbers (assuming space-separated)
                numbers = line.split()
        
                # Get the last 4 numbers
                last_four_numbers = numbers[1:]
        
                # Convert the strings to integers
                last_four_numbers = [float(num) for num in last_four_numbers]
        
                final.extend((last_four_numbers[0], last_four_numbers[1], last_four_numbers[2], last_four_numbers[3]))
        
        counter += 1

        print(final)

        pub_array = Float64MultiArray(data=final)
        self.msg_pub.publish(pub_array)



if __name__ == "__main__": 
    print("main")
    ROSBotDetectionNode()
    print("detection node")
