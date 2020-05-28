# std imports
import copy
import math
import re

# installed package imports
import pyrealsense2 as rs
import numpy as np
import cv2
from pyzbar import pyzbar


class rotation_estimator:
    def __init__(self):
        self.theta = [0.0, 0.0, 0.0]
        self.alpha = 0.98
        self.first = True
        self.last_ts_gyro = 0.0

    # Function to calculate the change in angle of motion based on data from gyro
    # gyro_data - list of 3 elems
    # ts - float arrival time of curr gyro frame
    def process_gyro(self, gyro_data, ts):
        if self.first:
            self.last_ts_gyro = ts
            return

        gyro_angle = [gyro_data.x, gyro_data.y, gyro_data.z]

        dt_gyro = (ts - self.last_ts_gyro) / 1000.0
        self.last_ts_gyro = ts

        gyro_angle = [gyro_angle[0] * dt_gyro, gyro_angle[1] * dt_gyro, gyro_angle[2] * dt_gyro]

        self.theta = [self.theta[0] - gyro_angle[2], self.theta[1] - gyro_angle[1], self.theta[2] + gyro_angle[0]]

    # Function to calculate the change in angle of motion based on data from accelerometer
    # accel_data - list of 3 elems
    def process_accel(self, accel_data):
        accel_angle = [0.0, 0.0, 0.0]
        accel_angle[2] = np.arctan2(accel_data.y, accel_data.z)
        accel_angle[0] = np.arctan2(accel_data.x, np.sqrt(accel_data.y ** 2 + accel_data.z ** 2))

        if self.first:
            self.first = False
            self.theta = accel_angle
            self.theta[1] = 0.0
        else:
            self.theta[0] = self.theta[0] * self.alpha + accel_angle[0] * (1 - self.alpha)
            self.theta[2] = self.theta[2] * self.alpha + accel_angle[2] * (1 - self.alpha)

    def get_theta(self):
        theta_out = [0.0, 0.0, 0.0]
        theta_out = [self.theta[0] * 180 / np.pi, self.theta[1] * 180 / np.pi,
                     (self.theta[2] - np.pi / 2) * 180 / np.pi]
        return theta_out


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # color stream
align = rs.align(rs.stream.color) # align both streams to same pov

# Enable visualizer and filters for later use
colorizer = rs.colorizer()
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)
spatial.set_option(rs.option.holes_fill, 3)
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Image detection size
expected = 300
inScaleFactor = 0.007843
meanVal = 127.53

net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')  # pretrained net

swapRB = True
classNames = { 0: 'background',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }


# State Representation
robot_x = 0
robot_y = 0
robot_theta = 0  # or 90?
robot_vx = 0
robot_vy = 0
human_x = 0
human_y = 0
human_theta = 0  # doesn't really matter
human_vx = 0
human_vy = 0
goal_x = 0  # set by user
goal_y = 0  # set by user
time_from_last_step = 0

# Start streaming
algo = rotation_estimator()
profile = pipeline.start(config)
first_run = True

try:
    while True:
        if first_run:
            try:
                frames = pipeline.wait_for_frames()
                ts = frames[1].as_motion_frame().get_timestamp()
                gyro_data = frames[1].as_motion_frame().get_motion_data()
                algo.process_gyro(gyro_data, ts)
                accel_data = frames[0].as_motion_frame().get_motion_data()
                algo.process_accel(accel_data)
                first_run = False
                continue
            except:
                continue

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        ts = frames[1].as_motion_frame().get_timestamp()
        gyro_data = frames[1].as_motion_frame().get_motion_data()
        algo.process_gyro(gyro_data, ts)
        accel_data = frames[0].as_motion_frame().get_motion_data()
        algo.process_accel(accel_data)

        if not depth_frame or not color_frame:
            # @TODO GET NEXT ACTION BASED ON JUST THE LAST STATE'S VALUES OR MAYBE STOP IN CURR POSITION
            continue

        # filter depth stream: depth2disparity -> spatial -> disparity2depth -> hole_filling
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # get intrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy arrays
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # detect and decode barcodes
        object_depth = np.asanyarray(depth_frame.get_data())
        barcode_poses = []
        barcodes = pyzbar.decode(color_image)

        bar_x = []
        bar_y = []
        robo_x = []
        robo_y = []

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(depth_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            curr = copy.deepcopy(object_depth)
            curr = curr[math.floor(((x) + (x + w)) / 2 - 1):math.ceil(((x) + (x + w)) / 2 + 1),
                        math.floor(((y) + (y + h)) / 2 - 1):math.ceil(((y) + (y + h)) / 2 + 1)].astype(float)
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            curr = curr * depth_scale
            dist, _, _, _ = cv2.mean(curr)
            barcode_poses.append((barcodeData, dist))

            # draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, dist)
            cv2.putText(color_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # store poses of codes
            curr_pose = re.split(',|\(|\)', barcodeData)
            bar_x.append(float(curr_pose[1]))
            bar_y.append(float(curr_pose[2]))

            # Get 3d point of object detected
            relative_depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                   [int((x + (x + w)) / 2), int((y + (y + h)) / 2)],
                                                                   dist)

            # Get world position of cam according to this barcode
            curr_cam_theta = algo.get_theta()[1]

            robot_theta = curr_cam_theta

            curr_cam_phi = np.arctan2(relative_depth_point[0], relative_depth_point[2]) * 180 / np.pi
            curr_cam_alpha = curr_cam_theta - curr_cam_phi
            curr_cam_delta_x = dist * np.cos(np.radians(curr_cam_alpha))
            curr_cam_delta_y = dist * np.sin(np.radians(curr_cam_alpha))
            robo_x.append(float(curr_pose[1] - curr_cam_delta_x))
            robo_y.append(float(curr_pose[2] - curr_cam_delta_y))

        if barcodes:
            curr_cam_pos = [np.average(robo_x), np.average(robo_y)]
            robot_x = curr_cam_pos[0]
            robot_y = curr_cam_pos[1]

            text_pos_x = 'X: ' + str(curr_cam_pos[0])
            text_pos_y = 'Y: ' + str(curr_cam_pos[1])

            cv2.putText(depth_image, text_pos_x, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(depth_image, text_pos_y, (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # FIND POSITION OF PERSON
        person_first_scan = []
        person_second_scan = []
        found_first = False
        found_second = False

        color_human_image = np.asanyarray(color_frame.get_data())

        # crop color image for detection
        height, width = color_human_image.shape[:2]
        expected = 300
        aspect = width / height
        resized_color_image = cv2.resize(color_human_image, (round(expected * aspect), expected))
        crop_start = round(expected * (aspect - 1) / 2)
        crop_color_img = resized_color_image[0:expected, crop_start:crop_start + expected]

        # Perform object detection through net
        blob = cv2.dnn.blobFromImage(crop_color_img, inScaleFactor, (expected, expected), meanVal, False)
        net.setInput(blob)
        detections = net.forward("detection_out")

        label = detections[0, 0, 0, 1]
        conf = detections[0, 0, 0, 2]
        xmin = detections[0, 0, 0, 3]
        ymin = detections[0, 0, 0, 4]
        xmax = detections[0, 0, 0, 5]
        ymax = detections[0, 0, 0, 6]

        if (conf >= .5):
            className = classNames[int(label)]

            # Calculate box coordinates of detected object
            scale = height / expected
            xmin_depth = int((xmin * expected + crop_start) * scale)
            ymin_depth = int((ymin * expected) * scale)
            xmax_depth = int((xmax * expected + crop_start) * scale)
            ymax_depth = int((ymax * expected) * scale)
            xmin_depth, ymin_depth, xmax_depth, ymax_depth

            # Calculate depth of object
            depth = np.asanyarray(depth_frame.get_data())
            # Crop depth data:
            depth = depth[math.floor((xmax_depth + xmin_depth) / 2 - 1):math.ceil((xmax_depth + xmin_depth) / 2 + 1),
                          math.floor((ymax_depth + ymin_depth) / 2 - 1):math.ceil((ymax_depth + ymin_depth) / 2 + 1)].astype(
                          float)

            # Get data scale from the device and convert to meters
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth = depth * depth_scale
            dist, _, _, _ = cv2.mean(depth)

            # Get relative location of person to camera
            relative_person_depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                          [int((xmin_depth + xmax_depth) / 2),
                                                                           int((ymin_depth + ymax_depth) / 2)],
                                                                          dist)

            # Get world position of cam according to this barcode
            curr_cam_theta = algo.get_theta()[1]
            curr_cam_phi = np.arctan2(relative_person_depth_point[0], relative_person_depth_point[2]) * 180 / np.pi
            curr_cam_alpha = curr_cam_theta - curr_cam_phi
            curr_person_delta_x = dist * np.cos(np.radians(curr_cam_alpha))
            curr_person_delta_y = dist * np.sin(np.radians(curr_cam_alpha))
            human_x = (float(robot_x + curr_person_delta_x))
            human_y = (float(robot_y + curr_person_delta_y))
            person_first_scan = [(float(robot_x + curr_person_delta_x)),
                                 (float(robot_y + curr_person_delta_y)),
                                 frames.get_timestamp()]
            found_first = True

            # Draw square on depth and color streams
            cv2.rectangle(depth_image, (xmin_depth, ymin_depth),
                          (xmax_depth, ymax_depth), (255, 255, 255), 2)
            cv2.rectangle(color_image, (xmin_depth, ymin_depth),
                          (xmax_depth, ymax_depth), (255, 255, 255), 2)
            cv2.putText(color_image, className + " @ " + "{:.2f}".format(dist) + "meters away",
                        (xmin_depth, ymin_depth),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

            # Stack both images horizontally
            images = np.hstack((color_image, depth_image))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        frame_count = 1
        while frame_count > 0:
            frame_count = frame_count - 1
            if found_second:
                break

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            ts = frames[1].as_motion_frame().get_timestamp()
            gyro_data = frames[1].as_motion_frame().get_motion_data()
            algo.process_gyro(gyro_data, ts)
            accel_data = frames[0].as_motion_frame().get_motion_data()
            algo.process_accel(accel_data)

            if not depth_frame or not color_frame:
                frame_count = frame_count - 1
                continue

            # filter depth stream: depth2disparity -> spatial -> disparity2depth -> hole_filling
            depth_frame = depth_to_disparity.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # get intrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

            # Convert images to numpy arrays
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_human_image = np.asanyarray(color_frame.get_data())

            # crop color image for detection
            height, width = color_human_image.shape[:2]
            expected = 300
            aspect = width / height
            resized_color_image = cv2.resize(color_human_image, (round(expected * aspect), expected))
            crop_start = round(expected * (aspect - 1) / 2)
            crop_color_img = resized_color_image[0:expected, crop_start:crop_start + expected]

            # Perform object detection through net
            blob = cv2.dnn.blobFromImage(crop_color_img, inScaleFactor, (expected, expected), meanVal, False)
            net.setInput(blob)
            detections = net.forward("detection_out")

            label = detections[0, 0, 0, 1]
            conf = detections[0, 0, 0, 2]
            xmin = detections[0, 0, 0, 3]
            ymin = detections[0, 0, 0, 4]
            xmax = detections[0, 0, 0, 5]
            ymax = detections[0, 0, 0, 6]

            if (conf >= .5):
                className = classNames[int(label)]

                # Calculate box coordinates of detected object
                scale = height / expected
                xmin_depth = int((xmin * expected + crop_start) * scale)
                ymin_depth = int((ymin * expected) * scale)
                xmax_depth = int((xmax * expected + crop_start) * scale)
                ymax_depth = int((ymax * expected) * scale)
                xmin_depth, ymin_depth, xmax_depth, ymax_depth

                # Calculate depth of object
                depth = np.asanyarray(depth_frame.get_data())
                # Crop depth data:
                depth = depth[math.floor((xmax_depth + xmin_depth) / 2 - 1):math.ceil((xmax_depth + xmin_depth) / 2 + 1),
                              math.floor((ymax_depth + ymin_depth) / 2 - 1):math.ceil((ymax_depth + ymin_depth) / 2 + 1)].astype(
                              float)

                # Get data scale from the device and convert to meters
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                depth = depth * depth_scale
                dist, _, _, _ = cv2.mean(depth)

                # Get relative location of person to camera
                relative_person_depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                              [int((xmin_depth + xmax_depth) / 2),
                                                                               int((ymin_depth + ymax_depth) / 2)],
                                                                              dist)

                # Get world position of cam according to this barcode
                curr_cam_theta = algo.get_theta()[1]
                curr_cam_phi = np.arctan2(relative_person_depth_point[0], relative_person_depth_point[2]) * 180 / np.pi
                curr_cam_alpha = curr_cam_theta - curr_cam_phi
                curr_person_delta_x = dist * np.cos(np.radians(curr_cam_alpha))
                curr_person_delta_y = dist * np.sin(np.radians(curr_cam_alpha))
                human_x = (float(robot_x + curr_person_delta_x))
                human_y = (float(robot_y + curr_person_delta_y))

                if not found_first:
                    person_first_scan = [(float(robot_x + curr_person_delta_x)),
                                         (float(robot_y + curr_person_delta_y)),
                                         frames.get_timestamp()]
                    found_first = True
                    frame_count = 1
                else:
                    person_second_scan = [(float(robot_x + curr_person_delta_x)),
                                          (float(robot_y + curr_person_delta_y)),
                                          frames.get_timestamp()]
                    found_second = True

                # Draw square on depth and color streams
                cv2.rectangle(depth_image, (xmin_depth, ymin_depth),
                              (xmax_depth, ymax_depth), (255, 255, 255), 2)
                cv2.rectangle(color_image, (xmin_depth, ymin_depth),
                              (xmax_depth, ymax_depth), (255, 255, 255), 2)
                cv2.putText(color_image, className + " @ " + "{:.2f}".format(dist) + "meters away",
                            (xmin_depth, ymin_depth),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

                # Stack both images horizontally
                images = np.hstack((color_image, depth_image))

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1)

                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

            frame_count = frame_count - 1

        if person_first_scan and person_second_scan:
            human_vx = (person_second_scan[0] - person_first_scan[0]) / (person_second_scan[2] - person_first_scan[2])
            human_vy = (person_second_scan[1] - person_first_scan[1]) / (person_second_scan[2] - person_first_scan[2])
            # @TODO get ORCA decision from robot_x/y, human_x/y, robot_vx/vy, human_vx/vy, robot_theta, & goal_x/y

        # else:
            # @TODO get ORCA decision from robot_x/y, robot_vx/vy, robot_theta, & goal_x/y (no human obstacle in observation)

        # @TODO set robot_vy/vy & theta in ROS, and act

finally:
    # Stop streaming
    pipeline.stop()
