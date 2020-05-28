# std imports
import copy
import math
import re

# installed package imports
import pyrealsense2 as rs
import numpy as np
import cv2
from pyzbar import pyzbar

# app module imports
from HumanDetector import HumanDetector
from RotationEstimator import RotationEstimator

class StateEstimator(object):
    def __init__(self):
        """
        Putpose:   Init state estimator object

        variables: pipeline - stream connection to camera
                   config - frame resolution & other configurations
                   align - aligns color & depth frames because they come from different places on camera
                   colorizer - converts depth frame to depth image
                   spatial, hole_filling, depth_to_disparity, disparity_to_depth - filters to clean up frames
                   human_det - object to detect humans and calculate their poses from a frame
                   rotation_est - object to calculate curr orientation of camera from gyro & accel data
                   x, y, vx, vy, theta - current pose & velocity of camera
                   profile - starts, stops, and grabs info about camera
        """
        # Configure streams to camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth, 640, 480, rs.format.z16, 30) # depth stream
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30) # color stream
        self.config.enable_stream(rs.stream.gyro) # gyro stream
        self.config.enable_stream(rs.stream.accel) # accelerometer stream
        self.align = rs.align(rs.stream.color) # align both depth & color streams to same pov

        # Enable visualizer and filters for later use
        self.colorizer = rs.colorizer()
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 50)
        self.spatial.set_option(rs.option.holes_fill, 3)
        self.hole_filling = rs.hole_filling_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        # Helper detection & calculation objects
        self.human_det = HumanDetector()
        self.rotation_est = RotationEstimator()

        # Pose of camera in world coordinates
        self.x = self.y = self.vx = self.vy = self.previous_time = 0.0
        self.theta = 90.0

        # Begin stream to camera
        self.profile = self.pipeline.start(self.config)


    def get_motion_frames(self, frames):
        """
        Putpose: Get the accelerometer & gyroscope frames & data from a list of all frames

        inputs:  frames - list of all frames received from camera (holds color, depth, gyro, & accel frames)

        Returns: None if gyro or accel frame was not received
                 Else gyro & accel data and timestamp

        Outputs: none
        """
        accel_data = gyro_data = gyro_ts = None

        # Grab accelerometer frame/data
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()

        # Grab gyroscope frame/data/timestamp
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gyro_ts = gyro_frame.as_motion_frame().get_timestamp()
        
        return accel_frame, accel_data, gyro_frame, gyro_data, gyro_ts


    def estimate_camera_position(
        self, color_image, depth_image, depth_frame, depth_intrin):
        """
        Putpose: Estimates x, y position of camera in world coordinates according
                 to detected barcodes in its frame

        inputs:  depth_frame - current depth frame of camera; use 'frames.get_depth_frame()'
                 depth_intrin - use 'depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics'
                 depth_image - depth image to draw detection box back onto
                 color_image - color image to draw detection box back onto

        Returns: None if no barcodes are detected in frame
                 Else x, y position of camera in world coordinates

        Outputs: Draws box around barcodes on depth_image & color_image if detected
        """
        # Detect and decode barcodes
        object_depth = np.asanyarray(depth_frame.get_data())
        barcodes = pyzbar.decode(color_image)

        # Vars to hold estimated poses of detected barcodes and camera
        barcode_poses = []
        bar_x = []
        bar_y = []
        robo_x = []
        robo_y = []

        # For each detected barcode in frame
        for barcode in barcodes:
            # Draw a box around the barcode on color & depth images
            (x, y, w, h) = barcode.rect
            cv2.rectangle(
                color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(
                depth_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Get encrypted data of position of barcode from image
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            # Get depth of current barcode and append it to barcode_poses list
            curr = copy.deepcopy(object_depth)
            curr = curr[
                math.floor(((x) + (x + w)) / 2 - 1) :
                    math.ceil(((x) + (x + w)) / 2 + 1), 
                math.floor(((y) + (y + h)) / 2 - 1) :
                    math.ceil(((y) + (y + h)) / 2 + 1)
            ].astype(float)
            depth_scale = self.profile.get_device(). \
                first_depth_sensor().get_depth_scale()
            curr = curr * depth_scale
            dist,_,_,_ = cv2.mean(curr)
            barcode_poses.append((barcodeData, dist))

            # Draw the barcode data and barcode type on the color image
            text = "{} ({})".format(barcodeData, dist)
            cv2.putText(
                color_image, text, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Append encrypted x, y of barcode to bar_x/y lists
            curr_pose = re.split(',|\(|\)',barcodeData)
            bar_x.append(float(curr_pose[1]))
            bar_y.append(float(curr_pose[2]))

            # Get 3d point of barcode detected relative to camera frame
            relative_depth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, 
                [int((x + (x + w)) / 2), int((y + (y + h)) / 2)], 
                dist)
            # print("\Relative Point: \nX ",relative_depth_point[0],"\nY ",relative_depth_point[1],"\nZ ",relative_depth_point[2])

            # Get world position of cam according to this barcode and append it to end of robo_x/y lists
            curr_cam_theta = self.theta
            # print("Cam Theta: \n",curr_cam_theta)
            curr_cam_phi = np.arctan2(
                relative_depth_point[2], relative_depth_point[0]) * 180 / np.pi
            # print("Cam Phi: \n",curr_cam_phi)
            curr_cam_alpha = 90 - curr_cam_theta + curr_cam_phi
            # print("Cam Alpha: \n",curr_cam_alpha)
            curr_cam_delta_x = dist * np.cos(np.radians(curr_cam_alpha))
            # print("Cam delta_x: \n",curr_cam_delta_x)
            curr_cam_delta_y = dist * np.sin(np.radians(curr_cam_alpha))
            # print("Cam delta_y: \n",curr_cam_delta_y)             
            # print("\nBarcode Pose: \nX ",curr_pose[1],"\nY ",curr_pose[2])
            robo_x.append(float(curr_pose[1]) - float(curr_cam_delta_x))
            robo_y.append(float(curr_pose[2]) - float(curr_cam_delta_y))

        # If barcodes were detected in the frame, return the average calculated position of the camera
        if barcodes:
            curr_cam_pos = [np.average(robo_x), np.average(robo_y)]

            # Draw estimated camera position text onto depth image & return estimate
            text_pos_x = 'X: ' + str(curr_cam_pos[0])
            text_pos_y = 'Y: ' + str(curr_cam_pos[1])
            cv2.putText(depth_image, text_pos_x, (20, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(depth_image, text_pos_y, (20, 420), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            return curr_cam_pos

        # No barcodes were detected, so we don't know the position of the camera from this frame
        return None


    def estimate_robot_state(self):
        """
        Putpose: Estimates robot_x, y, vx, vy, & theta values from 
                 current timestep and human_x, y, vx, vy if any are detected

        inputs:  none

        Returns: Observed position, orientation, & velocity of robot,
                 and position, & velocity of a human if any detected in camera view

        Outputs: Draws stream of depth and color images with estimated pose values
                 onto live OpenCV window
        """
        # If first run, calibrate gyroscope & accelerometer
        if self.rotation_est.first:
            frames = self.pipeline.wait_for_frames()
            accel_frame, accel_data, gyro_frame, gyro_data, gyro_ts = \
                self.get_motion_frames(frames)
            if not accel_frame or not gyro_frame:
                # print("FIRST FRAME NOT FOUND")
                return -1
            self.rotation_est.process_gyro(gyro_data, gyro_ts)
            self.rotation_est.process_accel(accel_data)
            self.previous_time = gyro_ts

        # Get current list of frames from camera
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        # get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            # print("DIDN'T GET DEPTH OR COLOR FRAME\n")
            return -1

        # Get motion frames, and calculate current orientation of camera
        accel_frame, accel_data, gyro_frame, gyro_data, gyro_ts = \
                self.get_motion_frames(frames)
        if gyro_frame:
            self.rotation_est.process_gyro(gyro_data, gyro_ts)
        if accel_frame:
            self.rotation_est.process_accel(accel_data)
        self.theta = self.rotation_est.get_theta()[1]

        # filter depth stream: depth2disparity -> spatial -> disparity2depth -> hole_filling
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        # get intrinsics of camera
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy arrays
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Estimate current x, y position & velocity of camera in world coords
        curr_cam_pos = self.estimate_camera_position(
            self.theta, color_image, depth_image, depth_frame, depth_intrin)
        if curr_cam_pos:
            self.vx = (curr_cam_pos[0] - self.x) / (gyro_ts - self.previous_time)
            self.vy = (curr_cam_pos[1] - self.y) / (gyro_ts - self.previous_time)
            self.x = curr_cam_pos[0]
            self.y = curr_cam_pos[1]
            self.previous_time = gyro_ts
        else:
            self.vx = 0
            self.vy = 0
            self.previous_time = gyro_ts

        # Estimate current x, y position of human in view in world coords
        curr_person_delta_x, curr_person_delta_y, curr_time = \
            self.human_det.detect(
            color_frame, depth_frame, self.profile, self.theta, 
            depth_intrin, depth_image, color_image)

        # Found a human, so try to calculate its velocity 
        if curr_person_delta_x is not None:
            new_frames = self.pipeline.wait_for_frames()
            new_frames = self.align.process(new_frames)

            # get depth and color frames
            new_depth_frame = new_frames.get_depth_frame()
            new_color_frame = new_frames.get_color_frame()

            # get intrinsics of camera
            new_depth_intrin = depth_frame.profile. \
                as_video_stream_profile().intrinsics

            # Convert images to numpy arrays
            new_depth_image = np.asanyarray(
                colorizer.colorize(new_depth_frame).get_data())
            new_color_image = np.asanyarray(
                new_color_frame.get_data())

            # Estimate new x, y position of human in view in world coords
            new_person_delta_x, new_person_delta_y, new_time = \
            self.human_det.detect(
                new_color_frame, new_depth_frame, self.profile, self.theta, 
                new_depth_intrin, new_depth_image, new_color_image)

            # If found human again, calculate its velocity
            if new_person_delta_x is not None:
                human_x = self.x + new_person_delta_x
                human_y = self.y + new_person_delta_y
                human_vx = (human_x - (self.x + curr_person_delta_x)) / \
                    (new_time - curr_time)
                human_vy = (human_y - (self.y + curr_person_delta_y)) / \
                    (new_time - curr_time)

            # If didn't find human again, assume static obstacle at previously detected location with no velocity
            else:
                human_x = self.x + curr_person_delta_x
                human_y = self.y + curr_person_delta_y
                human_vx = human_vy = 0.0

        # If no humans detected at all, then no human obstacle in current view
        else:
            human_vy = human_vy = human_x = human_y = None

        # Draw text with human pos and velocity onto image
        if human_x and human_y:
            text_pos_x = 'human X: ' + str(human_x)
            text_pos_y = 'human Y: ' + str(human_y)
            cv2.putText(depth_image, text_pos_x, (20, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(depth_image, text_pos_y, (20, 340), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if human_vx and human_vy:
            text_pos_x = 'human VX: ' + str(human_vx)
            text_pos_y = 'human VY: ' + str(human_vy)
            cv2.putText(depth_image, text_pos_x, (20, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(depth_image, text_pos_y, (20, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_image))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # Return observed state
        return self.x, self.y, self.vx, self.vy, self.theta, human_x, human_y, human_vx, human_vy

    def quit(self):
        """
        Putpose: Halts all streams running from camera

        inputs:  none

        Returns: none

        Outputs: Cleanly closes camera connection
        """
        self.pipeline.stop()
