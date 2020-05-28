# State Estimation on the D435i RealSense Camera
The goal of this project is to interface with the D435i RealSense camera to estimate the camera's pose and velocity at any timestep based on detected QR codes in its view, and if any humans are detected, also output their positions and velocities.

## Directory Overview
- main.py  
    - Example of a driver code which polls the camera, and prints out the state esimation, until the esc or q key is held down.
- StateEstimator.py
    - Holds class that estimates the pose of the camera & any humans it detects, and concatenates observation into a tuple.
- HumanDetector.py
    - Holds class that is used to detect humans in frames & output the offset of the human from the camera in the world coordinates.
- RotationEstimator.py
    - Holds class that is used to calculate the current theta orientation of the robot in the world coordinates.
- frozen_inference_graph.pb & graph.pbtxt
    - Pretrained ssd mobilenet used to detect humans in images.
- prototype_jupyter_notebooks/
    - A directory holding Jupyter Notebooks with incremental changes as I got closer to concatenating the state estimation. The latest working notebook is titled 'Part 7 - Cleaning Up State Information.ipynb'.
- tutorials/
    - Tests written in notebooks to become familiar with the RealSense camera.
- reading_resources/
    - Holds some papers that helped me along the way.
- QR_Code_Locations.zip
    - Zipped folder holding PNGs of QR codes encoded with x, y, z position in meters. The name of each file is the same as the respective code's enocded position in meters.

## Dependencies  
The versions of installed python modules that have this code working on my local machine are as follows:
- python - 3.7.3
- numpy - 1.18.1
- pyzbar - 0.1.8
- pyrealsense2 - 2.32.1.1299
- opencv - 4.1.2

## Running Example  
1. Print out as many QR codes as needed, and place them on the floor in the real world depending on their encoded positions. For example, the `0.5,2,0` QR code should be 0.5 meters to the right & 2 meters ahead of the `0,0,0` QR code.  
2. Run `python main.py` from main directory.  
3. View the camera stream on the laptop as the camera is moved in different locations around the floor. 

## Future Goals
The camera localization is dependent on physically placed QR codes in the real world, which are prone to degredation over time, or may be missed by the view of the camera. I'd like to implement a SLAM algorithm which will first map the robot's surroundings into a point cloud, and then given a goal, find a path to that goal while checking for humans and obstacles along the way. Most of the implementations I have found use the D435i & T265 cameras in conjunction, but here are a few tutorials that use only the D435i I found that seem ok in my opinion:
- https://shinkansan.github.io/2019-UGRP-DPoom/SLAM
- https://github.com/IntelRealSense/realsense-ros/wiki/SLAM-with-D435i
