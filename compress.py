from PIL import Image
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

def compress(img_msg):
    cv_image = None
     # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = CvBridge().imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    color_converted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color_converted)
 
    # Resize the image
    image = image.resize((38,28))
    image.save('0.jpg')

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)