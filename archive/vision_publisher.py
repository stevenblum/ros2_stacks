#!/usr/bin/env python3
from MCMOTracker import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import PoseStamped
import tf2_ros
import geometry_msgs.msg
import random
from time import time

ARM_CAMERA_IMAGE_TOPIC = '/camera/color/image_raw'
MO
CONFIDENCE_THRESH

class TilePublisher(Node):
    def __init__(self):
        super().__init__('tile_publisher')

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            ARM_CAMERA_IMAGE_TOPIC,  # topic name
            self.image_callback,
            1  # queue size
        )
        self.subscription  # prevent unused variable warning

        self.w2r_calibration_status = False
        self.w2r_calibration_time = None

        self.mcmot = MCMOTracker("LATEST",[0,1],["lab_logitech1","lab_logitech2"], "4square")

        self.loop_timer_cb = self.create_timer(0.5, self.loop_timer_cb)

    def image_callback(self,msg):
        # Convert ROS Image message → OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')



    def timer_callback(self):
        # Example dummy tile detection

        self.mcmot.detect_track_and_annotate()


        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.pose.position.x = 0.25 + random.uniform(-0.05, 0.05)
        pose.pose.position.y = 0.10 + random.uniform(-0.05, 0.05)
        pose.pose.position.z = 0.02
        pose.pose.orientation.w = 1.0
        self.publisher_.publish(pose)
        self.get_logger().info(f'Published tile pose: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')


    def pick_and_stack():


    def calibrate_w2r(self):
        continue_cal = True

        camera_3dp_robot = []
        camera_3dp_world = []

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        aruco_config = MCCOTracker.ArucoConfig("square4")
                
        while continue_cal:

            # OpenCV IM Show the Image, self.frame is being updated by image_callback
            cv2.imshow("Arm Camera", self.frame)

            # Wait for key stroke "c"

            # If Keystroke "c"

                # Look for all aruco markers in image

                # If less than 4 aruco markers found

                    #Print less than 4
                
                    # Continue 

                # Loop through aruco markers and get image coordinates and world cordinates from dict

                # Run Open cv2.PNPSolve to determin pose of the camera

                # Append the "Pose" 3d position of camera in world coordinates in camera_3dp_world

                # Use ROS2 TF2 to determine location camera in robot coordinates and append to camera_3dp_robot

                # Is "q" key is hit, continue_cal=False

        # Print Status Message, creating 

        # Create a rotation and tranlsation matrix that will transfrom 3d points from world to robot coordinates
        # Kabsch algorithm or Horn's method
        # Find Centroid of Both Sets of Points

        # Trnaslate Points to Origin

        # Compute Covariance Matrix

        # Perform Singular Value Decomposition

        # Calculate the Rotation Matrix

        # Check if the Determinant of R is -1, if so reflection, multiply v3 by -1

        # Calculate the Translation Vector

        # Form the homogenous transofrmation matrix



def main(args=None):
    rclpy.init(args=args)
    node = TilePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
