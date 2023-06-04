#!/usr/bin/env python
# The line above is important so that this file is interpreted with Python when running it.

# Author: Jake Markus
# Date: April 3 2023

# Import of python modules.
import math  # use of pi.
from compress import compress
import rospy  # module for ROS APIs
from geometry_msgs.msg import Twist  # message type for cmd_vel
from sensor_msgs.msg import LaserScan  # message type for scan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
import cv2

# Frequency at which the loop operates
FREQUENCY = 10  # Hz.

# Velocities that will be used (feel free to tune)
LINEAR_VELOCITY = .1  # m/s
ANGULAR_VELOCITY = math.pi / 4  # rad/s

MIN_SCAN_ANGLE_RAD = -20.0 / 180 * math.pi
MAX_SCAN_ANGLE_RAD = +20.0 / 180 * math.pi

MIN_THRESHOLD_DISTANCE = 1 # m, threshold distance, should be smaller than range_max

class Pfield:
    def __init__(self, resolution, obstacleLocation, size):
        self.resolution = resolution
        self.size = size
        self.obstacleLocation = obstacleLocation


    def distance(self, q1,q2,rr=0,so=0):
        return round(np.linalg.norm(q2 - q1), 2) - rr - so

    def uAttractive(self,z, q, q0):
        d=self.distance(q,q0)
        return (((z * (d ** 2) * 1 / 2)))

    def fAttractive(self,z, q, q0):
        return ((z * (q - q0)))


    def uRepulsive(self, n, q_star, q, oq, rr, so):
        r = []
        for array in (oq.T):
            array = array.reshape(2, 1)
            d = self.distance(q, array, rr, so)
            if d <= q_star:
                r.append((0.5 * n * ((1 / d) - (1 / q_star)) ** 2))
            else:
                r.append(0)
        return sum(r)


    def fRepulsive(self, n, q_star, q, oq, rr, so):
        r = np.zeros((2, 1))
        for array in oq.T:
            array = array.reshape(2, 1)
            d = self.distance(q, array, rr, so)

            if d <= q_star:
                r = np.append(r, (n * ((1 / q_star) - (1 / d)) * ((q - (array)) / (d ** 3))), axis=1)
            else:
                r = np.append(r, np.zeros((2, 1)), axis=1)

        return np.sum(r, axis=1).reshape(2, 1)


    def GradientDescent(self, q, oq, q0, alpha, max_iter, n, q_star, zeta,  U_star, rr, so):
        success = False
        U = self.uRepulsive(n, q_star, q, oq, rr, so) + self.uAttractive(zeta, q, q0)
        U_hist = [U]
        q_hist = q
        for i in range(max_iter):

            if U > U_star:
                grad_total = self.fRepulsive(n, q_star, q, oq, rr, so) + self.fAttractive(zeta, q, q0)
                q = q - alpha * (grad_total / np.linalg.norm(grad_total))
                U = self.uRepulsive(n, q_star, q, oq, rr, so) + self.uAttractive(zeta, q, q0)
                q_hist = np.hstack((q_hist, q))
                U_hist.append(U)
              
            else:
                print("Algorithm converged successfully and Robot has reached goal location")
                break
            if i == max_iter - 1:
                print("Robot is either at local minima or loop ran out of maximum  iterations")

        return q_hist, U_hist

    def demo(self,start,goal):

        # resolution = 10
        start = np.array([start[0]*10, start[1]]*10)
        start.resize((2, 1))
        obstacleLocation = np.array([self.obstacleLocation[0]*self.resolution, self.obstacleLocation[1]*self.resolution])
        obstacleLocation.resize((2, 1))
        goal = np.array([goal[0]*10, goal[1]*10+1])
        goal.resize((2, 1))

        max_iter = 2000
        alpha = 0.1
        n = 1
        zeta = 1
        U_star = 0.1
        q_star = 1
        robot_radius = 10
        obstacle_size = 5
        q_hist, U_hist = self.GradientDescent(
            start, obstacleLocation, goal, alpha, max_iter, n, q_star, zeta, U_star, robot_radius, obstacle_size)

        path = q_hist
        return path

class PID:

    def __init__(self, kp, ki, kd, k):
        """
        :param kp: The proportional gain constant
        :param ki: The integral gain constant
        :param kd: The derivative gain constant
        :param k: The offset constant that will be added to the sum of the P, I, and D control terms
        """
        self._p = kp
        self._i = ki
        self._d = kd
        self._k = k
        self._err_prev = None
        self._err_sum = 0

    def step(self, err, dt):
        """
        This method will get called at each sensor update, and returns/calculates the PID command

        :param err: The current error (difference between the setpoint and drone's current altitude) in meters.
                    For example, if the drone was 10 cm below the setpoint, err would be 0.1
        :param dt: The time (in seconds) elapsed between measurements of the process variable (the drone's altitude)
        :returns: command
        """

        u = 0
        self._err_sum += err
        if self._err_prev:
            u = self._p * err + self._d * (err - self._err_prev) / dt + self._i * dt * self._err_sum + self._k

        self._err_prev = err

        return u

    def reset(self):
        """
        Resest the PID terms so that previously stored values will
        not affect the current calculations.
        """
        self._err_prev = None
        self._err_sum = 0

"""A class to handle position readings. Should probably be changed"""
class Radar:
    def __init__(self, pose, time_a):
        self.pose = pose
        self.aq_time = time_a

"""A class to handle our object avoider, which has everything internally."""
class ObjectAvoider():
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY,scan_angle=[MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD],
                 min_threshold_distance=MIN_THRESHOLD_DISTANCE,):
        """Constructor."""

        # Setting up publishers/subscribers.
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        # Setting up subscriber receiving messages from the laser.

        rospy.Subscriber("odom", Odometry, self.odom_callback)

        # Parameters.
        self.linear_velocity = linear_velocity  # Constant linear velocity set.
        self.angular_velocity = angular_velocity  # Constant angular velocity set.

        self.REACT_DIST = 2.5 #HEURISTICS
        self.DODGE_DIST = 1.25

        self.PID = PID(-5.0, 0.0, -3.5, 0.0)
        self.REACHED_THRESHOLD = 0.5
        # Flag used to control the behavior of the robot.
        self.pose = (0, 0, 0) #Current pose of the robot: x, y, theta. Should immediately overwritten by odom
        self._laser_sub = rospy.Subscriber("scan", LaserScan, self._laser_callback, queue_size=1)

        self.ml_labels = []
        self.enemyreadings = []
        self.personalreadings = []
        self.scan_angle = scan_angle
        self.min_threshold_distance = min_threshold_distance

    def image_callback(self, msg):
        image = compress(msg)
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
        print("the labels: ", self.ml_labels)

    def _laser_callback(self, msg):
        """Processing of laser message."""

        if len(self.ml_labels) != 0:
            self.obstacle_angle_to_base_link = self.getObstacleAngle(self.ml_labels)
            x,y = self.detectObstaclePosition(self.obstacle_angle_to_base_link, msg)
            enemy_pos = (x,y)
            reading = Radar(enemy_pos, rospy.get_rostime())
            self.enemyreadings.append(reading)
        else:
            # Calculate indexe range based off angle and increment
            min_index = (int)(0 / msg.angle_increment)
            max_index = (int)((2 * math.pi) / msg.angle_increment)

            min_range = msg.ranges[
                max_index-1]  # initial value is arbitrary, but range is upper bound exclusive so including it here
            theta = math.pi
            selected_theta = math.pi
            # Loop through every other bounded index
            for i in range(min_index, max_index):

                # If within scanner range and less than the current min, update it
                if msg.ranges[i] < min_range:
                    min_range = msg.ranges[i]
                    selected_theta = theta
                print(msg.ranges[i])
                theta += msg.angle_increment

            # Raise the flag if within threshold dist
            enemy_pos = (
            self.pose[0] + math.cos(selected_theta) * min_range, self.pose[1] + math.sin(selected_theta) * min_range)
            reading = Radar(enemy_pos, rospy.get_rostime())
            self.enemyreadings.append(reading)
            print("enemy position" + str(enemy_pos))

    def getObstacleAngle(self, positions):
        # positions given as x, y, width, height
        dist_to_mid_line = 15 - (positions[0]+(positions[3]/2.0)) # x position of the incoming robot
        angle_direction = dist_to_mid_line/16.0 * (60.0/2.0)
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

    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)

    def rotAngle(self, angle, rate):

        rotate_time = float(abs(angle)) / ANGULAR_VELOCITY
        start_time = rospy.get_rostime()
        # marks start_time to calculate duration

        # spin until spin_time is reached
        while rospy.get_rostime() - start_time < rospy.Duration(rotate_time):
            self.move(0, math.copysign(ANGULAR_VELOCITY, angle))
            rate.sleep()

        self.stop()

    def rotAngleAtSpeed(self, angle, rate):

        rotate_time = float(abs(angle)) / ANGULAR_VELOCITY
        start_time = rospy.get_rostime()
        # marks start_time to calculate duration

        # spin until spin_time is reached
        while rospy.get_rostime() - start_time < rospy.Duration(rotate_time):
            self.move(LINEAR_VELOCITY, math.copysign(ANGULAR_VELOCITY, angle))
            rate.sleep()

    def moveDist(self, dist, rate):

        move_time = float(abs(dist))/LINEAR_VELOCITY
        start_time = rospy.get_rostime()
        # marks start_time to calculate duration

        # spin until spin_time is reached
        while rospy.get_rostime() - start_time < rospy.Duration(move_time):
            self.move(math.copysign(LINEAR_VELOCITY, dist), 0)
            rate.sleep()
        self.stop()

    def getDist(self, pose1, pose2):
        return math.sqrt((pose2[0]-pose1[0])**2 +(pose2[1]-pose1[1])**2 )

    """Calculates the intersection of two lines based on two points from each"""
    def lineLineIntersection(self, A, B, C, D):
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1 * (A[0]) + b1 * (A[1])

        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2 * (C[0]) + b2 * (C[1])

        determinant = a1 * b2 - a2 * b1

        if (determinant == 0):
            return None

        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return (x, y)

    """Moves to a point in a straight line until too close to an opponent on an intersecting path"""
    def smartMoveToPoint(self, x, y, rate):

        pf = Pfield(10, (100, 100), 10)
        q_hist = pf.demo((self.pose[0], self.pose[1]), (x, y))
        points = []
        for i in range(0, len(q_hist[0])):
            points.append([q_hist[0][i] / 10.0, q_hist[1][i] / 10.0])
        points.append([x, y])
        print(points)


        print("YEE")
        print("Points: " + str(points[0]))
        last_time_stamp = rospy.get_time()
        self.PID.reset()

        xdiff = x - self.pose[0]
        ydiff = y - self.pose[1]

        theta = math.atan2(ydiff, xdiff)  # calculate desired angle

        # How much we need to rotate and move
        anglediff = theta - self.pose[2]

        # speeds up movement by not rotating miniscule amounts
        if abs(abs(anglediff) - 2 * math.pi) < 0.1 or abs(anglediff) < 0.1:
            anglediff = 0.0

        # send commands
        if anglediff != 0.0:
            self.rotAngleAtSpeed(anglediff, rate)

        while abs(self.pose[0] - x) > self.REACHED_THRESHOLD or abs(self.pose[1] -y)  > self.REACHED_THRESHOLD:


            if len(self.enemyreadings) < 2 or self.getDist(self.pose, self.enemyreadings[-1].pose) >= self.REACT_DIST:

                error = 100000
                best_dist = 10000
                goodpoint = []

                for point in points:

                    if self.getDist(point, self.pose) <= best_dist:
                        error = self.getDist(self.pose, point) * math.cos(math.atan2(self.pose[0] - point[0], self.pose[1] - point[1]) - self.pose[2])
                        best_dist = self.getDist(point, self.pose)
                        goodpoint = point



                print(str(error) + "," + str(goodpoint) + ", " + str(self.pose))
                result = self.PID.step(error, float(rospy.get_time()-last_time_stamp))
                last_time_stamp = rospy.get_time()
                self.move(LINEAR_VELOCITY, result)
                rate.sleep()
            else:
                intersect = self.lineLineIntersection(self.enemyreadings[-2].pose, self.enemyreadings[-1].pose, self.personalreadings[-2].pose, self.personalreadings[-1].pose)

                if not intersect == None and self.getDist(self.pose, intersect) <= self.DODGE_DIST:
                    print("intersect is " +str(intersect))
                    enemy_xdiff = (self.enemyreadings[-2].pose[0] - self.enemyreadings[-1].pose[0] * 1.0 )
                    enemy_ydiff = (self.enemyreadings[-2].pose[1] - self.enemyreadings[-1].pose[1] * 1.0 )

                    vector = [0.0, enemy_ydiff/abs(enemy_ydiff)] if enemy_xdiff == 0.0 else [1.0, enemy_ydiff/enemy_xdiff]
                    vector = [float(i)/sum(vector) for i in vector]
                    print(vector)
                    vector = [vector[0] * self.DODGE_DIST, vector[1] * self.DODGE_DIST]

                    target = (intersect[0] + vector[0], intersect[1] + vector[1])

                    self.moveToPoint(target[0], target[1], rate)
                    self.smartMoveToPoint(x, y, rate) # Call this function again to go back to the original target
                    return #End the current instance of this function, no recursion here

                else:
                    error = 100000
                    best_dist = 10000
                    goodpoint = []

                    for point in points:

                        if self.getDist(point, self.pose) <= best_dist:
                            error = self.getDist(self.pose, point) * math.cos(
                                math.atan2(self.pose[0] - point[0], self.pose[1] - point[1]) - self.pose[2])
                            best_dist = self.getDist(point, self.pose)
                            goodpoint = point

                    print(str(error) + "," + str(goodpoint) + ", " + str(self.pose))
                    result = self.PID.step(error, float(rospy.get_time() - last_time_stamp))
                    last_time_stamp = rospy.get_time()
                    self.move(LINEAR_VELOCITY, result)
                    rate.sleep()


        print("REACHED!")

    def moveToPoint(self, x, y, rate):
        """Move robot to point in global_pos reference frame by Rotation then Translation"""

        xdiff = x - self.pose[0]
        ydiff = y - self.pose[1]

        theta = math.atan2(ydiff, xdiff)  # calculate desired angle

        # How much we need to rotate and move
        anglediff = theta - self.pose[2]
        movedist = math.sqrt(xdiff ** 2 + ydiff ** 2)

        # speeds up movement by not rotating miniscule amounts
        if abs(abs(anglediff) - 2 * math.pi) < 0.1 or abs(anglediff) < 0.1:
            anglediff = 0.0

        # send commands
        self.rotAngleAtSpeed(anglediff, rate)
        self.moveDist(movedist, rate)
        self.moveDist(movedist, rate)

    def drawPolygon(self, points, rate):
        #Move to every point in the polyline
        for point in points:
            self.smartMoveToPoint(point[0], point[1], rate)
        self.stop()

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def odom_callback(self, msg):
        quaterion = msg.pose.pose.orientation
        quaterion_list = [quaterion.x, quaterion.y, quaterion.z, quaterion.w] # Extract rotation from odom

        euler_z = euler_from_quaternion(quaterion_list)[2] #exctract euler z from quaternion conversion
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, euler_z) #Put in correct format for pose
        reading = Radar((msg.pose.pose.position.x, msg.pose.pose.position.y), rospy.get_rostime())
        self.personalreadings.append(reading)


def main():
    """Main function."""

    # 1st. initialization of node.
    rospy.init_node("Collision_Ideal")
    rate = rospy.Rate(FREQUENCY)

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(5)

    # Initialization of the class for the random walk.
    dodger = ObjectAvoider()

    # If interrupted, send a stop command before interrupting.
    rospy.on_shutdown(dodger.stop)

    # Robot random walks.
    try:
        dodger.drawPolygon([(3,0), (3, 1)], rate)

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()
