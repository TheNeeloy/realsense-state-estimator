# std imports
import math

# installed package imports
import pyrealsense2 as rs
import numpy as np
import cv2

class HumanDetector(object):
    def __init__(self):
        """
        Putpose:   Init human detector object

        variables: expected, in_scale_factor, mean_val - size vars used for running MobileNet SSD Model
                   net - pretrained MobileNet SSD Model
                   swap_RB - bool to say if swapping first and last channels in 3-channel image is necessary
                   class_names - labels that can be detected by MobileNet SSD Model
        """
        # Image detection size
        self.expected = 300
        self.in_scale_factor = 0.007843
        self.mean_val = 127.53
        self.swap_RB = False

        # Load net
        self.net = cv2.dnn.readNetFromTensorflow(
            'frozen_inference_graph.pb', 'graph.pbtxt')
        self.class_names = { 0: 'background',
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


    def detect(self, color_frame, depth_frame, profile, robot_theta,
               depth_intrin, depth_image, color_image):
        """
        Putpose: Detects if a human is present in current frame and returns offset
                 of human from camera in world coordinates

        inputs:  color_frame - current color frame of camera; use 'frames.get_color_frame()'
                 depth_frame - current depth frame of camera; use 'frames.get_depth_frame()'
                 profile - camera stats; originally set from 'profile = pipeline.start(config)'
                 robot_theta - euler angle of robot in world coordinates
                 depth_intrin - use 'depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics'
                 depth_image - depth image to draw detection box back onto
                 color_image - color image to draw detection box back onto

        Returns: None if no humans are detected in frame
                 Else x, y offset of human from camera & timestamp of when human was detected

        Outputs: Draws box around human on depth_image & color_image if detected
        """
        color_human_image = np.asanyarray(color_frame.get_data())
    
        # crop color image for detection
        height, width = color_human_image.shape[:2]
        aspect = width / height
        resized_color_image = cv2.resize(
            color_human_image, (round(self.expected * aspect), 
            self.expected))
        crop_start = round(self.expected * (aspect - 1) / 2)
        crop_color_img = resized_color_image[
            0:self.expected, crop_start:crop_start + self.expected]

        # Perform object detection through net
        blob = cv2.dnn.blobFromImage(
            crop_color_img, self.in_scale_factor, 
            (self.expected, self.expected), self.mean_val, self.swap_RB)
        self.net.setInput(blob)
        detections = self.net.forward("detection_out")

        # Extract output from net
        label = detections[0, 0, 0, 1]
        conf = detections[0, 0, 0, 2]
        xmin = detections[0, 0, 0, 3]
        ymin = detections[0, 0, 0, 4]
        xmax = detections[0, 0, 0, 5]
        ymax = detections[0, 0, 0, 6]

        # If detected human in frame
        if (conf >= .5):
            className = self.class_names[int(label)]

            # Calculate box coordinates of detected object
            scale = height / self.expected
            xmin_depth = int((xmin * self.expected + crop_start) * scale)
            ymin_depth = int((ymin * self.expected) * scale)
            xmax_depth = int((xmax * self.expected + crop_start) * scale)
            ymax_depth = int((ymax * self.expected) * scale)
            xmin_depth, ymin_depth, xmax_depth, ymax_depth

            # Calculate depth of object
            depth = np.asanyarray(depth_frame.get_data())

            # Crop depth data:
            depth = depth[
                math.floor((xmax_depth + xmin_depth) / 2 - 1) : 
                    math.ceil((xmax_depth + xmin_depth) / 2 + 1), 
                math.floor((ymax_depth + ymin_depth) / 2 - 1) :
                    math.ceil((ymax_depth + ymin_depth) / 2 + 1)
            ].astype(float)

            # Get data scale from the device and convert to meters
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth = depth * depth_scale
            dist, _, _, _ = cv2.mean(depth)

            # Get relative location of person to camera
            relative_depth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [int((xmin_depth + xmax_depth) / 2), 
                int((ymin_depth + ymax_depth) / 2)], dist)

            # Get x, y offset of human from robot in world coordinates
            curr_cam_theta = robot_theta
            # print("Cam Theta: \n",curr_cam_theta)
            curr_cam_phi = np.arctan2(
                relative_depth_point[2], relative_depth_point[0]) * 180 / np.pi
            # print("Cam Phi: \n",curr_cam_phi)
            curr_cam_alpha = 90 - curr_cam_theta + curr_cam_phi
            # print("Cam Alpha: \n",curr_cam_alpha)
            curr_person_delta_x = dist * np.cos(np.radians(curr_cam_alpha))
            # print("person delta_x: \n", curr_person_delta_x)
            curr_person_delta_y = dist * np.sin(np.radians(curr_cam_alpha))
            # print("person delta_y: \n", curr_person_delta_y)

            # Draw square on depth and color streams where human was detected
            cv2.rectangle(
                depth_image, (xmin_depth, ymin_depth), 
                (xmax_depth, ymax_depth), (255, 255, 255), 2)
            cv2.rectangle(
                color_image, (xmin_depth, ymin_depth), 
                (xmax_depth, ymax_depth), (255, 255, 255), 2)
            cv2.putText(
                color_image, 
                className + " @ " + "{:.2f}".format(dist) + "meters away", 
                (xmin_depth, ymin_depth), cv2.FONT_HERSHEY_COMPLEX, 
                0.5, (255, 255, 255))

            # Return detected human offset from camera in world coordinates
            return curr_person_delta_x, curr_person_delta_y, color_frame.get_timestamp()
        
        # No humans detected
        return None, None, None
