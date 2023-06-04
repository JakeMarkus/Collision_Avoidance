import numpy as np
import matplotlib.pyplot as plt

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
                if i % 25 == 0:
                    print(f"Potential after {i} iterations is " + str(U))
                #     print(q)
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

        fig, axes = plt.subplots()
        for ob in (obstacleLocation.T):
            ob = ob.reshape(1, 2)
            plt.plot(ob[0, 0], ob[0, 1], marker="o")
            circle = plt.Circle((ob[0, 0], ob[0, 1]), obstacle_size, fill=True)
            circle2 = plt.Circle((ob[0, 0], ob[0, 1]), obstacle_size + q_star, fill=False)
            axes.set_aspect(1)
            axes.add_artist(circle)
            axes.add_artist(circle2)

        for point in (q_hist.T):
            point = point.reshape(1, 2)
            plt.plot(point[0, 0], point[0, 1], marker=".")
            circle3 = plt.Circle((point[0, 0], point[0, 1]), robot_radius, fill=False)
            axes.add_artist(circle3)

        plt.plot(start[0, 0], start[1, 0], marker="o", color="red")
        plt.plot(goal[0, 0], goal[1, 0], marker="o", color="red")

        plt.title("Path followed by the Robot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.ylim(-100,100)
        plt.xlim(0,200)
        plt.grid()
        plt.show()

        path = q_hist
        print(path)
        return path

if __name__ == '__main__':
    obstacleLocation = (5,0)
    Pf = Pfield(10,obstacleLocation,10)
    Pf.demo((0,0),(8,0))
