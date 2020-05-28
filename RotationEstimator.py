# installed package imports
import numpy as np

class RotationEstimator(object):
    def __init__(self):
        """
        Putpose:   Init rotation estimator object

        variables: theta - x, y, z orientation of cam in world
                   alpha - weight used in accelerometer data processing
                   first - boolean of first processing run or not
                   last_ts_gyro - timestamp of last gyroscope processing
        """
        self.theta = [0.0, 0.0, 0.0]
        self.alpha = 0.98
        self.first = True
        self.last_ts_gyro = 0.0


    def process_gyro(self, gyro_data, ts):
        """
        Putpose: Calculate the change in angle of motion based on data from gyro

        inputs:  gyro_data - list of 3 elems from rs.stream.gyro
                 ts - float arrival time of curr gyro frame
                 
        Returns: none

        Outputs: Updates x, y, z rotation of self.theta according to gyro data
        """
        if self.first:
            self.last_ts_gyro = ts
            return
        
        gyro_angle = [gyro_data.x, gyro_data.y, gyro_data.z]
        
        dt_gyro = (ts - self.last_ts_gyro) / 1000.0
        self.last_ts_gyro = ts
        
        gyro_angle = [gyro_angle[0] * dt_gyro, 
                      gyro_angle[1] * dt_gyro, 
                      gyro_angle[2] * dt_gyro]
        
        self.theta = [self.theta[0] - gyro_angle[2], 
                      self.theta[1] - gyro_angle[1], 
                      self.theta[2] + gyro_angle[0]]


    def process_accel(self, accel_data):
        """
        Putpose: Calculate the change in angle of motion based on data from accelerometer

        inputs:  accel_data - list of 3 elems from rs.stream.accel
                 ts - float arrival time of curr accel frame

        Returns: none

        Outputs: Updates x, y, z rotation of self.theta according to accel data
        """
        accel_angle = [0.0, 0.0, 0.0]
        accel_angle[2] = np.arctan2(accel_data. y,accel_data.z)
        accel_angle[0] = np.arctan2(
            accel_data.x, np.sqrt(accel_data.y**2 + accel_data.z**2))
        
        if self.first:
            self.first = False
            self.theta = accel_angle
            self.theta[1] = np.pi / 2
        else:
            self.theta[0] = self.theta[0] * self.alpha + \
                accel_angle[0] * (1 - self.alpha)
            self.theta[2] = self.theta[2] * self.alpha + \
                accel_angle[2] * (1 - self.alpha)


    def get_theta(self):
        """
        Putpose: Return the current euler angle orientation of the robot in the world frame

        inputs:  none

        Returns: self.theta x, y, z orientation in degrees
        
        Outputs: none
        """
        theta_out = [0.0, 0.0, 0.0]
        theta_out = [self.theta[0] * 180 / np.pi, 
                     self.theta[1] * 180 / np.pi, 
                     (self.theta[2] - np.pi / 2) * 180 / np.pi]
        return theta_out
