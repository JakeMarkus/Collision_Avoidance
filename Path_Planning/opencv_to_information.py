#!/usr/bin/env python

# Author:Ivy Zhang, Hongke Lu, Jake Markus
# Date: May 31st

# Import of python modules.
import math # use of pi.
import numpy as np # use of some math
import random
# import of relevant libraries.
import rospy # module for ROS APIs
from PIL import Image
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist # message type for cmd_vel
from nav_msgs.msg import Odometry # message from odom
from sensor_msgs.msg import LaserScan # message type for scan
from compress import compress
from std_msgs.msg import Float64MultiArray
import cv2

import tf # used for some transformation

# Constants.
# Topic names
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_ODOM_TOPIC = '/odom'
DEFAULT_SCAN_TOPIC = '/scan' # you need to change it to '/scan' if you want to use rosbot
                            # or base_scan if working on other robots
DEFAULT_MAP_TOPIC = '/map'
DEFAULT_MAP_SIZE = 600 #assume square map
DEFAULT_RESOLUTION = 20 #20 grids per meter

# Field of view in radians that is checked in front of the robot (feel free to tune)
MIN_SCAN_ANGLE_RAD = -20.0 / 180 * math.pi
MAX_SCAN_ANGLE_RAD = +20.0 / 180 * math.pi


# Frequency at which the loop operates
FREQUENCY = 10 #Hz.

# Velocities that will be used (feel free to tune)
LINEAR_VELOCITY = 0.2 # m/s
ANGULAR_VELOCITY = math.pi/8 # rad/s
MIN_THRESHOLD_DISTANCE = 1 # m, threshold distance, should be smaller than range_max


class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        # Here we want to apply a blur algorithm to make the wall bigger
        self.resolution = resolution

    def cell_at(self, x, y):
        #find the value of tha cell
        x = int(x)
        y = int(y)
        return self.grid[y, x]

    def cell_free(self,x,y):
        # mark the cell as free cell
        x = int(x)
        y = int(y)
        self.grid[y, x] = 0

    def cell_occupied(self,x,y):
        # mark the cell as occpied cell. 
        x = int(x)
        y = int(y)
        self.grid[y, x] = 100

class Information_Extraction:

    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY, resolution = DEFAULT_RESOLUTION, 
                 min_threshold_distance=MIN_THRESHOLD_DISTANCE,
                 map_size = DEFAULT_MAP_SIZE,
                 scan_angle=[MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD]):
        """Constructor."""

        # Setting up publishers/subscribers for ML Rosbot Detection
        self._img_sub = rospy.Subscriber('/camera/rgb/image_rect_color', Image, self._image_callback, queue_size=1)
        self._laser_sub = rospy.Subscriber(DEFAULT_SCAN_TOPIC, LaserScan, self._laser_callback, queue_size=1)
        # self._detect_pub = rospy.Publisher("rosbot_detection", Float64MultiArray, queue_size=10)

        # Setting up publishers/subscribers for INFORMATION EXTRACTION
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = rospy.Publisher(DEFAULT_CMD_VEL_TOPIC, Twist, queue_size=1) #publish the control
        self._map_pub = rospy.Publisher(DEFAULT_MAP_TOPIC, OccupancyGrid, queue_size = 1)
        self._odom_ = rospy.Subscriber(DEFAULT_ODOM_TOPIC,Odometry,self._updatePose) #subscribe to the odom

        # Mathematical Parameters
        self.linear_velocity = linear_velocity # Constant linear velocity set.
        self.angular_velocity = angular_velocity # Constant angular velocity set.
        self.map_size = map_size
        self.resolution = resolution
        self.x = 0 # monitor the current location x
        self.y = 0 # monitor the current location y
        self.quaternion =[0,0,0,0] #save the quaternion
        self.targetPoints = [] #save the target points 
        self.angle = 0 # current orientation of the robot respect to the odom frame
        self.local_A_odom = 0 # initialize the transformation matrix
        self.lastGrid = Grid(np.full(self.map_size*self.map_size, -1), self.map_size, self.map_size, DEFAULT_RESOLUTION)
        self.currentGrid = Grid(np.full(self.map_size*self.map_size, -1), self.map_size, self.map_size, DEFAULT_RESOLUTION)
        # initialize the map using occupancy grid
        self.header = Header(stamp=rospy.Time.now(), frame_id=DEFAULT_ODOM_TOPIC)
        self.map_meta_data = MapMetaData(rospy.Time(0), 1.0/self.resolution,
                                         self.map_size, self.map_size,
                                         Pose(Point(-self.map_size/self.resolution/2, -self.map_size/self.resolution/2, 0), Quaternion()))
        self.map = OccupancyGrid(self.header, self.map_meta_data, np.full(self.map_size*self.map_size, -1))
        self.listener = tf.TransformListener()


        ### ML Object Detection / Information Extraction Parameters
        self.min_threshold_distance = min_threshold_distance
        self.scan_angle = scan_angle
        # Lidar state params
        self.laser_scan_msg = None
        self._close_obstacle = False
        self.ml_labels = []
        self.enemy_robot_position = []

    def _image_callback(self, msg):
        print("Hellow")
        image = compress(msg)
        print("Hellow2")
        # OpenCV opens images as BRG; but we want it as RGB We'll also need a grayscale version
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use minSize because for not
        # bothering with extra-small
        # dots that would look like STOP signs
        stop_data = cv2.CascadeClassifier('cascade.xml')

        found = stop_data.detectMultiScale(img_gray,
                                           minSize=(20, 20))

        label_coordinates = []

        for (x, y, width, height) in found:
            label_coordinates.extend((x, y, width, height))

        # print(label_coordinates)
        self.ml_labels = label_coordinates
        print("Here's the label: ", self.ml_labels)

        pub_array = Float64MultiArray(data=label_coordinates)
        self._detect_pub.publish(pub_array)


    def _laser_callback(self, msg):
        """Processing of laser message."""
        print("laser scanning....")
        # we want to keep track of the forward distance and the angle of the min distance angle
        self.laser_scan_msg = msg
        self.forwardDistance = msg.ranges[len(msg.ranges) - 1]
        self.minDistanceAngle = msg.angle_max - (
                    len(msg.ranges) - msg.ranges.index(min(msg.ranges))) * msg.angle_increment
        self.createGrid()
        print("Grid created")
        self.publishOccupancyGrid()
        self.lastGrid = self.currentGrid

        print("getting information, ")
        self.obstacle_angle_to_base_link = self.getObstacleAngle(self.ml_labels)
        self.detectObstaclePosition(self.obstacle_angle_to_base_link, msg)
        print("here here here")

    def _updatePose(self,msg):
        self.x = msg.pose.pose.position.x # update current x
        self.y = msg.pose.pose.position.y # update current y
        self.quaternion[0] = msg.pose.pose.orientation.x # update current quaternion x
        self.quaternion[1] = msg.pose.pose.orientation.y # update current quaternion y
        self.quaternion[2] = msg.pose.pose.orientation.z # update current quaternion z
        self.quaternion[3] = msg.pose.pose.orientation.w # update current quaternion w
        # update current angle 
        angle = tf.transformations.euler_from_quaternion(self.quaternion, axes='sxyz')[2]
        if angle <=0:
            self.angle = 2*math.pi + angle
        else:
            self.angle = angle
        
    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        return self._cmd_pub.publish(twist_msg)

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)
     
    def rotate_rel(self,theta):
        """This function will calculate the duration to rotate for theta 
        and move the robot for theta"""
        if abs(theta) > math.pi:
            if theta > 0:
                theta = math.pi*2-theta
            if theta < 0:
                theta = math.pi*2+theta
        duration = theta/self.angular_velocity
        # if we have negative angle then we should rotate to the other direction 
        if duration < 0:
            duration = -duration
            currentVelocity = -self.angular_velocity
        else:
            currentVelocity = self.angular_velocity
        start_time = rospy.get_rostime()
        rate = rospy.Rate(FREQUENCY) 
        while rospy.get_rostime() - start_time < rospy.Duration(duration):
        # let the robot rotate
            self.move(0,currentVelocity)
            rate.sleep()
        self.stop()

    def translate(self,d):
        """This function will calculate the duration to move for d
        and move the robot for d"""
        duration = d/self.linear_velocity
        rate = rospy.Rate(FREQUENCY) 
        start_time = rospy.get_rostime()
        while rospy.get_rostime() - start_time < rospy.Duration(duration):
            self.move(self.linear_velocity,0)
            rate.sleep()
        self.stop()

    def calculateDistance(self,x,y):
        """This function will calculate the distance between two points"""
        return math.sqrt((x-self.x)**2+(y-self.y)**2)

    def createGrid(self):  
        # if we have laser scan msg
        if self.laser_scan_msg:
           
            start_angle = self.angle + self.laser_scan_msg.angle_min #- math.pi
            center = DEFAULT_MAP_SIZE/2 - 1 #center of the map
            for i in range(len(self.laser_scan_msg.ranges)):
                current_angle = start_angle + self.laser_scan_msg.angle_increment*i
                # if the laser scan if within the range of the lidar
                if self.laser_scan_msg.range_min < self.laser_scan_msg.ranges[i] < self.laser_scan_msg.range_max:
                    x = int(center + (self.x + math.cos(current_angle)*self.laser_scan_msg.ranges[i]) * self.resolution)
                    y = int(center + (self.y + math.sin(current_angle)*self.laser_scan_msg.ranges[i]) * self.resolution)
                    # we want to ray tracing of the lidar
                    self.rayTracing(center,center,x,y)
    
    def publishOccupancyGrid(self):
        # create the occupancy grid and publish it
        occupancyGrid = OccupancyGrid()
        occupancyGrid.info = self.map_meta_data
        occupancyGrid.data = np.reshape(self.currentGrid.grid, self.map_size*self.map_size)
        occupancyGrid.header.stamp = rospy.Time.now()
        occupancyGrid.header.frame_id = "odom"
        self._map_pub.publish(occupancyGrid)
    
    def rayTracing(self, x0, y0, x1, y1): #from wikipedia
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            if DEFAULT_MAP_SIZE-1 > x0 > 0 and DEFAULT_MAP_SIZE-1 > y0 > 0:
                self.currentGrid.cell_free(x0, y0)
                if x0 == x1 and y0 == y1:
                    self.currentGrid.cell_occupied(x0, y0)
                    break
                e2 = 2 * error
                if e2 >= dy:
                    if x0 == x1:
                        break
                    error += dy
                    x0 += sx
                if e2 <= dx:
                    if y0 == y1:
                        break
                    error += dx
                    y0 += sy
            else:
                break


    def getObstacleAngle(self, positions):
        # positions given as x, y, width, height
        dist_to_mid_line = 15 - (positions[0]+(positions[3]/2.0)) # x position of the incoming robot
        print(positions)
        # print("distance to mid line: ", dist_to_mid_line)
        angle_direction = dist_to_mid_line/19.0 * (60.0/2.0)
        print("direction of enemy robot to base link: ", angle_direction)
        return angle_direction


    def detectObstaclePosition(self, angle, msg):
        print(msg.angle_min, msg.angle_max)
        min_index = int(round((self.scan_angle[0] + angle - msg.angle_min + math.pi) / msg.angle_increment))
        # calculate the max index
        max_index = int(round(min_index + (self.scan_angle[1] + angle - self.scan_angle[0] + math.pi) / msg.angle_increment))
        print(min_index, max_index)
        # parse the ranges to what we set
        view_range = msg.ranges[min_index:max_index]
        # print("view ranges: ", view_range)

        if min(view_range) < self.min_threshold_distance:
            # set up the flag
            self._close_obstacle = True

        index_robot = view_range.index(min(view_range))
        obstacle_angle = msg.angle_min + index_robot * msg.angle_increment + math.pi
        print("obstacle angle: ", obstacle_angle)
        y = -math.sin(obstacle_angle) * min(view_range)
        x = math.cos(obstacle_angle) * min(view_range)
        print("position of enemy robot to base link: ", (x, y))
        self.enemy_robot_position = [x, y]
        return x, y


    def getObstacleVelocity(self, list_of_positions):
        list_of_velocities = []

        for order in range(1, len(list_of_positions)):
            current_position = list_of_positions[order]
            last_position = list_of_positions[order-1]

            time = 10
            euclidean_distance = math.sqrt((current_position[1]-last_position[1])**2
                                        +(current_position[0]-last_position[0])**2)

            velocity = euclidean_distance/time
            list_of_velocities.append(velocity)

        return list_of_velocities


    def spin(self):
        rate = rospy.Rate(FREQUENCY) # loop at 10 Hz.
        while not rospy.is_shutdown():
            print("random walking")
            print(self.enemy_robot_position)
            # if not self._close_obstacle:
            #     # go straigh a head
            #     self.move(self.linear_velocity,0)
            # else:
            #     # get current time
            #     start_time = rospy.get_rostime()
            #     # rotate random angle between -pi and pi
            #     angle = random.uniform(-1,1)*math.pi
            #     # calculate the delta t
            #     duration = abs(angle/self.angular_velocity)
            #     #decide the direction of turn
            #     if angle<0:
            #         current_angular_velocity = self.angular_velocity*-1
            #     else:
            #         current_angular_velocity = self.angular_velocity
            #     # let the robot move for delta t
            #     while rospy.get_rostime() - start_time <= rospy.Duration(duration):
            #         # let the robot rotate
            #         self.move(0,current_angular_velocity)
            #     #reset the flag
            #     self._close_obstacle = False
            # ####### ANSWER CODE END #######
            rate.sleep()
       

def main():
    """Main function."""

    # 1st. initialization of node.
    rospy.init_node("InformationExtraction")

    # Initialization of the class for the random walk.
    pa4 = Information_Extraction()

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)

    # If interrupted, send a stop command before interrupting.
    rospy.on_shutdown(pa4.stop)

    # Robot random walks.
    try:
        pa4.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()
