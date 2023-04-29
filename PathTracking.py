############################################################
# LOWER LAYER
############################################################

# Readme
# Simulation: run this file using coderunner or in the terminal
#

import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import Spline
import cv2

class State():
    # Hold state variables
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        # STATE
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.delta = 0

        # VEHICLE
        self.a = 2  # [cm]: D(wheeltocenter)
        self.l = 4  # [cm]: D(wheeltowheel)

    # Update state
    def update(self, acceleration, delta):
        # Clip delta between min and max steering
        self.delta = np.clip(delta, -self.max_steer, self.max_steer)
        # STATE SPACE MODEL
        self.x += self.v*np.cos(self.delta)*np.cos(self.yaw)*self.dt
        self.y += self.v*np.cos(self.delta)*np.sin(self.yaw)*self.dt
        self.yaw -= (self.v/self.l)*np.sin(self.delta)*self.dt 
        self.yaw = self.normalize_angle(self.yaw)
        self.v += acceleration * self.dt
    
    def normalize_angle(self, angle):
        # Rescale angle to [-pi,pi]
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

class PathTracking(State):

    def __init__(self, xd, yd, yaw, show_animation=True):
        # PID
        self.Kp = 7
        self.Kd = 0.5
        self.prev_err = 0   

        # STANLEY STEERING
        self.k = 6  # control gain
        self.prev_err_stan = 0

        # SIMULATION
        self.dt = 0.05      # [s] time difference
        self.max_sim_time = 100
        self.show_animation = show_animation

        # VEHICLE CONSTRAINTS
        self.max_steer = np.deg2rad(45) # max steering angle
        self.target_speed = 25          # [cm/s]

        # VARIA
        self.accel_distance = 7
        self.slow_distance = 15     # [cm]

        # Spline initialization
        self.px, self.py, self.pyaw, self.pk, self.s = Spline.calc_spline_course(xd, yd, ds=0.05)
        self.pv = self.calculateSpeed()

        # Create state object
        State.__init__(self, xd[0], yd[0], self.normalize_angle(yaw), 0)

        # Calculate path
        self.rx, self.ry, self.ryaw, self.rv, self.rt, self.rs = self.calculatePath()

    def stanleyControl(self, previousindex):
        # Determine index of nearest points
        currentindex, error = self.nearestIndex()
        # Don't turn back
        if previousindex >= currentindex:
            currentindex = previousindex
        # Correct heading error
        theta_e = self.normalize_angle(self.yaw - self.pyaw[currentindex])
        # Correct cross track error
        theta_d = -np.arctan2(self.k * error, self.v)
        # Steering control
        err_stan = theta_e + theta_d
        delta = 1.8 * err_stan + 0.1 * (err_stan - self.prev_err_stan)/self.dt
        self.prev_err_stan = err_stan
        # Return steeringangle and current index
        return delta, currentindex
    
    def nearestIndex(self):
        # Search nearest point index
        dx = [self.x - icx for icx in self.px]
        dy = [self.y - icy for icy in self.py]
        d = np.hypot(dx, dy)        # return sqrt(dx**2+dy**2)
        targetindex = np.argmin(d)  # index of min distance
        # Determine front axle orientation
        front_axle_vec = [np.sin(self.yaw), -np.cos(self.yaw)]
        # Project error on front axle
        error_front_axle = np.dot([dx[targetindex], dy[targetindex]], front_axle_vec)
        # Return values
        return targetindex, error_front_axle
    
    # PD-controller for speed
    def pid_control(self, target, current):
        # Error
        err = target - current
        # Control value
        out = self.Kp * err + self.Kd * (err - self.prev_err)/self.dt
        # Assign new prev error
        self.prev_err = err
        # Return control value
        return out

    # Calculate speed profile (slow down in sharp turns and at the end)
    def calculateSpeed(self):
        # Target speed overall
        speed = [self.target_speed] * len(self.s)
        # Find relative extrema using scipy
        rel_extr_index = argrelextrema(np.abs(self.pk), np.greater)
        # Loop over extrema
        for i in rel_extr_index[0]:
            # Set threshold
            if np.abs(self.pk[i]) > 0.07:
                # Sharper turn -> slow down earlier
                scale = np.interp(np.abs(self.pk[i]), (0.05, 0.6), (30, 250))
                magn = np.interp(np.abs(self.pk[i]), (0.05, 0.6), (0.05, 0.6))
                # Mask for speed profile
                speedmask = 1-magn*np.exp(((-1)*(self.s-self.s[i])**2)/scale)
                # Multiply mask with speed
                speed = np.multiply(speed, speedmask)
        # Speed up at start
        accelindex = np.argmax(np.array(self.s)>=self.accel_distance)
        for i in range(accelindex):
            speed[i] = np.interp(i, (0, accelindex), (0, speed[accelindex]))
        # Speed down at the end
        slowindex = np.argmax(np.array(self.s)>=(self.s[-1]-self.slow_distance))
        for i in range(slowindex, len(self.s)):
            speed[i] = np.interp(i, (slowindex, len(self.s)-1), (speed[slowindex], 0))
        # Minimum speed to prevent long end
        speed = np.where(np.array(speed) < 3, 3, np.array(speed))
        # Return speed profile
        return speed

    def calculatePath(self):
        # Stop condition
        stopindex = len(self.px)-1
        time = 0.0
        # Output
        x = [self.x]
        y = [self.y]
        deltalist = [0]
        yaw = [self.yaw]
        v = [self.v]
        t = [0.0]
        s = [0]
        targetindex, _ = self.nearestIndex()
        # Simulation
        while time <= self.max_sim_time and targetindex < stopindex:
            # Calculate acceleration via PID control
            accel = self.pid_control(self.pv[targetindex], self.v)
            # Calculate steering angle with stanley
            delta, targetindex = self.stanleyControl(targetindex)
            # Update states
            self.update(accel, delta)
            # Update time
            time += self.dt
            # Update outputs
            x.append(self.x)
            y.append(self.y)
            deltalist.append(self.delta)
            yaw.append(self.yaw)
            v.append(self.v)
            t.append(time)
            s.append(self.s[targetindex])
            # Animation
            if self.show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(self.px, self.py, ".r", label="Reference")
                plt.plot(x, y, "-b", label="Trajectory")
                plt.plot(self.px[targetindex], self.py[targetindex], "xg", label="Target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[cm/s]: " + str(self.v)[:4] + " Time[s]: " + f"{time:.2f}")
                # plt.title("Delta" + str(np.rad2deg(self.delta)) + "MaxDelta" + str(np.rad2deg(max(deltalist))))
                plt.pause(0.001)
        # Return ouputs
        return x,y,yaw,v,t,s
    
    def showPath(self):
        # Plot output
        plt.subplots(1)
        plt.plot(self.px, self.py, 'r--', label="Reference")
        plt.plot(self.rx, self.ry, 'g--', label="Robot")
        plt.legend()
        plt.xlabel("X[m]")
        plt.ylabel("Y[m]")
        plt.axis("equal")
        plt.grid(True)

        # Plot velocity
        plt.subplots(1)
        plt.plot(self.s, self.pv, 'r--', label="Reference")
        plt.plot(self.rs, self.rv, 'g--', label="Velocity")
        plt.legend()
        plt.xlabel("T[s]")
        plt.ylabel("V[cm/s]")
        plt.axis("equal")
        plt.grid(True)

        # Plot kromming
        plt.subplots(1)
        plt.plot(self.s, np.abs(self.pk), 'r--', label="Kromming")
        plt.legend()
        plt.xlabel("S[cm]")
        plt.ylabel("K")
        plt.grid(True)
        plt.show()

    def drawPath(self, img, rr):
        image = img.copy()
        # self.showPath()
        for i in range(len(self.rx)-1):
            image = cv2.line(image, (int(self.rx[i]*rr), int(self.ry[i]*rr)), (int(self.rx[i+1]*rr), int(self.ry[i+1]*rr)), (220, 220, 220), 15)
        for i in range(len(self.rx)-1):
            image = cv2.line(image, (int(self.rx[i]*rr), int(self.ry[i]*rr)), (int(self.rx[i+1]*rr), int(self.ry[i+1]*rr)), (0, 0, 200), 2)
        # Return values
        return image, self.ryaw[-1], self.rs[-1]

# Simulation
if __name__ == '__main__':
    # Function to track
    x = [ 0, 74, 74, 60,  0,  6,  7, 34, 42,  0]
    y = [80, 70, 24, 70, 70, 58, 37, 44, 23, 16]
    # Spline
    test = PathTracking(x, y, 0)
    test.showPath()


