#!/usr/bin/env python
# The line above is important so that this file is interpreted with Python when running it.

# Author: Jake Markus
# Date: April 3 2023

# Import of python modules.
import math  # use of pi.
import random  # used to generate random spin time
import threading
# import of relevant libraries.
import time
import numpy

import rospy  # module for ROS APIs
from geometry_msgs.msg import Twist  # message type for cmd_vel
from sensor_msgs.msg import LaserScan  # message type for scan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np

# Frequency at which the loop operates
FREQUENCY = 10  # Hz.

# Velocities that will be used (feel free to tune)
LINEAR_VELOCITY = .5  # m/s
ANGULAR_VELOCITY = math.pi / 4  # rad/s

OFFSET = [5.0, 5.0]
SPACE = 1.0




def distance(q1,q2,rr=0,so=0):
    return round(np.linalg.norm(q2 - q1), 2) - rr - so

def uAttractive(z, q, q0):
    d=distance(q,q0)
    return (((z * (d ** 2) * 1 / 2)))

def fAttractive(z, q, q0):
    return ((z * (q - q0)))


def uRepulsive(n, q_star, q, oq, rr, so):
    r = []
    for array in (oq.T):
        array = array.reshape(2, 1)
        d = distance(q, array, rr, so)
        if d <= q_star:
            r.append((0.5 * n * ((1 / d) - (1 / q_star)) ** 2))
        else:
            r.append(0)
    return sum(r)


def fRepulsive(n, q_star, q, oq, rr, so):
    r = np.zeros((2, 1))
    for array in oq.T:
        array = array.reshape(2, 1)
        d = distance(q, array, rr, so)
        if d <= q_star:
            r = np.append(r, (n * ((1 / q_star) - (1 / d)) * ((q - (array)) / (d ** 3))), axis=1)
        else:
            r = np.append(r, np.zeros((2, 1)), axis=1)

    return np.sum(r, axis=1).reshape(2, 1)


def GradientDescent(q, oq, q0, alpha, max_iter, n, q_star, zeta,  U_star, rr, so):
    success = False
    U = uRepulsive(n, q_star, q, oq, rr, so) + uAttractive(zeta, q, q0)
    U_hist = [U]
    q_hist = q
    for i in range(max_iter):

        if U > U_star:
            grad_total = fRepulsive(n, q_star, q, oq, rr, so) + fAttractive(zeta, q, q0)
            q = q - alpha * (grad_total / np.linalg.norm(grad_total))
            U = uRepulsive(n, q_star, q, oq, rr, so) + uAttractive(zeta, q, q0)
            q_hist = np.hstack((q_hist, q))
            U_hist.append(U)
            # if i % 25 == 0:
            #     print(f"Potential after {i} iterations is " + str(U))
            #     # print(q)
        else:
            #print("Algorithm converged successfully and Robot has reached goal location")
            break
        # if i == max_iter - 1:
        #     #print("Robot is either at local minima or loop ran out of maximum  iterations")

    return q_hist



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

"""A simple class for a target robot that moves for a bit"""
class EnemyRobot():
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY,):
        """Constructor."""

        # Setting up publishers/subscribers.
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = rospy.Publisher("/robot_1/cmd_vel", Twist, queue_size=1)
        # Setting up subscriber receiving messages from the laser.

        rospy.Subscriber("/robot_1/odom", Odometry, self.odom_callback)

        # Parameters.
        self.linear_velocity = linear_velocity  # Constant linear velocity set.
        self.angular_velocity = angular_velocity  # Constant angular velocity set.

        # Flag used to control the behavior of the robot.
        #self._close_obstacle = False  # Flag variable that is true if there is a close obstacle.
        self.pose = (0, 0, 0) #Current pose of the robot: x, y, theta. Should immediately overwritten by odom

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

    def moveDist(self, dist, rate):

        move_time = float(abs(dist))/LINEAR_VELOCITY
        start_time = rospy.get_rostime()
        # marks start_time to calculate duration

        # spin until spin_time is reached
        while rospy.get_rostime() - start_time < rospy.Duration(move_time):
            self.move(math.copysign(LINEAR_VELOCITY, dist), 0)
            rate.sleep()
        self.stop()

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
        self.rotAngle(anglediff, rate)
        self.moveDist(movedist, rate)

    def drawPolygon(self, points, rate):
        #Move to every point in the polyline
        for point in points:
            self.moveToPoint(point[0], point[1], rate)

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def odom_callback(self, msg):

        quaterion = msg.pose.pose.orientation
        quaterion_list = [quaterion.x, quaterion.y, quaterion.z, quaterion.w] # Extract rotation from odom

        euler_z = euler_from_quaternion(quaterion_list)[2] #exctract euler z from quaternion conversion
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, euler_z) #Put in correct format for pose
        #print(self.pose)

    def trapezoid(self, radius, rate):

        ##### Using Draw Polygon ######

        top_base_val = radius * math.cos(math.pi / 4) # The x and y coords of the leg verts with be the same value because of the 45 degree angle

        points = [(0, radius), (top_base_val, top_base_val), (top_base_val, -1 * top_base_val), (0, -1 * radius),
                  (0, 0)] # The point to visit in order, ending with returning to (0,0)

        #Transforms the points so that the trapezoid is relative to the robots position
        current_pos = (self.pose[0], self.pose[1])
        for i in range(0, len(points)):
            points[i] = (points[i][0] + current_pos[0], points[i][1] + current_pos[1])

            print(points[i])

        #Passes all the points to the polygon function
        self.drawPolygon(points, rate)
"""A class to handle position readings. Should probably be changed"""
class Radar:
    def __init__(self, pose, time_a):
        self.pose = pose
        self.aq_time = time_a

"""A class to handle our object avoider, which has everything internally."""
class ObjectAvoider():
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY,):
        """Constructor."""

        # Setting up publishers/subscribers.
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = rospy.Publisher("/robot_0/cmd_vel", Twist, queue_size=1)
        # Setting up subscriber receiving messages from the laser.

        rospy.Subscriber("/robot_0/odom", Odometry, self.odom_callback)

        # Parameters.
        self.linear_velocity = linear_velocity  # Constant linear velocity set.
        self.angular_velocity = angular_velocity  # Constant angular velocity set.

        self.REACT_DIST = 2.5 #HEURISTICS
        self.DODGE_DIST = 1.25

        self.PID = PID(-5.0, 0.0, -3.5, 0.0)
        self.REACHED_THRESHOLD = 0.5
        # Flag used to control the behavior of the robot.
        #self._close_obstacle = False  # Flag variable that is true if there is a close obstacle.
        self.pose = (0, 0, 0) #Current pose of the robot: x, y, theta. Should immediately overwritten by odom
        rospy.Subscriber("/robot_1/odom", Odometry, self.calc_enemy_position)
        self.enemyreadings = []
        self.personalreadings = []

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
            # print("NONING!")
            # print(str(A) + ", " + str(B) + ", " + str(C) + ", " + str(D))
            return None

        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return (x, y)

    def potentialMoveToPoint(self, x, y, rate):
        env_size = 1000
        q = np.array([self.pose[0], self.pose[1]])
        q = np.vstack(q)
        oq = np.array([[100], [100]])
        q0 = np.array([x, y])
        q0 = np.vstack(q0)

        # print("Randomly generated start position is \n " + str(q))
        # print("Randomly generated Goal position is \n" + str(q0))
        # print("Randomly generated obstacles are \n" + str(oq))

        max_iter = 2000
        alpha = 0.1
        n = 1
        zeta = 1
        U_star = 0.1
        q_star = 1
        d_star = 5
        robot_radius = 1
        obstacle_size = 1
        q_hist = GradientDescent(
            q, oq, q0, alpha, max_iter, n, q_star, zeta, U_star, robot_radius, obstacle_size)

        points = []
        for i in range(0, len(q_hist[0])):
            points.append([q_hist[0][i], q_hist[1][i]])
        points.append([x, y])

        print("YEE")
        # print(q_hist)
        print(points)
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


            error = 100000
            best_dist = 10000
            goodpoint = []
            # for point in points:
            #     if self.getDist(self.pose, point) <= error:
            #         error = self.getDist(self.pose, point)
            #
            #         if self.pose[1] < point[1] and error < 5:
            #             error *= -1
            #         goodpoint = point

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

    """Moves to a point in a straight line until too close to an opponent on an intersecting path"""
    def smartMoveToPoint(self, x, y, rate):

        env_size = 1000
        q = np.array([self.pose[0], self.pose[1]])
        q = np.vstack(q)

        oq = np.array([[], []])
        q0 = np.array([x, y])
        q0 = np.vstack(q0)

        # print("Randomly generated start position is \n " + str(q))
        # print("Randomly generated Goal position is \n" + str(q0))
        # print("Randomly generated obstacles are \n" + str(oq))

        max_iter = 2000
        alpha = 0.1
        n = 1
        zeta = 1
        U_star = 0.1
        q_star = 1
        d_star = 5
        robot_radius = 1
        obstacle_size = 1
        q_hist = GradientDescent(
            q, oq, q0, alpha, max_iter, n, q_star, zeta, U_star, robot_radius, obstacle_size)

        points = []
        for i in range(0, len(q_hist[0])):
            points.append([q_hist[0][i], q_hist[1][i]])
        points.append([x, y])

        print("YEE")
        #print(q_hist)
        print(points)
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

            if self.getDist(self.pose, self.enemyreadings[-1].pose) >= self.REACT_DIST:

                error = 100000
                best_dist = 10000
                goodpoint = []
                # for point in points:
                #     if self.getDist(self.pose, point) <= error:
                #         error = self.getDist(self.pose, point)
                #
                #         if self.pose[1] < point[1] and error < 5:
                #             error *= -1
                #         goodpoint = point

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
                    # for point in points:
                    #     if self.getDist(self.pose, point) <= error:
                    #         error = self.getDist(self.pose, point)
                    #
                    #         if self.pose[1] < point[1] and error < 5:
                    #             error *= -1
                    #         goodpoint = point

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

    def calc_enemy_position(self, msg):

        quaterion = msg.pose.pose.orientation
        quaterion_list = [quaterion.x, quaterion.y, quaterion.z, quaterion.w]  # Extract rotation from odom

        euler_z = euler_from_quaternion(quaterion_list)[2]  # exctract euler z from quaternion conversion
        reading = Radar((-1 * msg.pose.pose.position.y + OFFSET[0], -1 * msg.pose.pose.position.x + OFFSET[1]), rospy.get_rostime())
        self.enemyreadings.append(reading)
        #print("Enemy: " + str(reading.pose[0]) + ", " + str(reading.pose[1]))

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
        #print("Self: " + str(self.pose))

    def trapezoid(self, radius, rate):

        ##### Using Draw Polygon ######

        top_base_val = radius * math.cos(math.pi / 4) # The x and y coords of the leg verts with be the same value because of the 45 degree angle

        points = [(0, radius), (top_base_val, top_base_val), (top_base_val, -1 * top_base_val), (0, -1 * radius),
                  (0, 0)] # The point to visit in order, ending with returning to (0,0)

        #Transforms the points so that the trapezoid is relative to the robots position
        current_pos = (self.pose[0], self.pose[1])
        for i in range(0, len(points)):
            points[i] = (points[i][0] + current_pos[0], points[i][1] + current_pos[1])

            print(points[i])

        #Passes all the points to the polygon function
        self.drawPolygon(points, rate)

def main():
    """Main function."""

    # 1st. initialization of node.
    rospy.init_node("Collision_Ideal")
    rate = rospy.Rate(FREQUENCY)

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(5)

    # Initialization of the class for the random walk.
    enemy = EnemyRobot()
    dodger = ObjectAvoider()

    # If interrupted, send a stop command before interrupting.
    rospy.on_shutdown(enemy.stop)
    rospy.on_shutdown(dodger.stop)

    t1 = threading.Thread(target=dodger.drawPolygon, args=([(8,0), (0,0)], rate))
    t2 = threading.Thread(target=enemy.moveDist, args=(7, rate))

    #rospy.sleep(5)
    # Robot random walks.
    try:
        rospy.sleep(1)
        # enemy.moveDist(5, rate)
        # dodger.moveDist(5, rate)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()
