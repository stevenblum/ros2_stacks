
    def transform_w2r(world_3dp):
        # multiple by self.R_w2r and self.t_w2r

        # Return 3D point in Robot Frame


    def calibrate_w2r(self):
        continue_cal = True

        arm_camera_3dp_robot = []
        arm_camera_3dp_world = []

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        aruco_config = MCCOTracker.ArucoConfig("square4")  # keys=aruco markers numbers, values= 3d point in world coordinates

        # Setup temporary node, that will allow me to get the cameras 3d coordinates in the robots frame inside this function
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, node)

        # Allow some time for TF messages to populate
        self.get_logger().info("Waiting for TF to fill...")
        rclpy.spin_once(self, timeout_sec=1.0)

                
        while continue_cal:
            # OpenCV IM Show the Image, self.frame is being updated by image_callback
            cv2.imshow("Arm Camera", self.frame)

            # Wait for key stroke "c"

            # If Keystroke "c"

                # Look for all aruco markers in image

                # If less than 4 aruco markers found

                    # Print less than 4 markers found
                
                    # Continue 

                aruco_2dp_image = []
                aruco_known_3dp_w = []

                # For Loop: each marker found in the image

                    # If Aruco marker in aruco_config.keys()

                        # Append 3dp from aruco_config{marker number} to aruco_known_3dp_w

                    # Else
                        # return WARNING, "node.calibrate_w2r() found aruco marker not in aruco_config"

                        # Continue

                    # Append image coodinates to aruco_camera_2dp

                # If less than 4 points

                # Run Open cv2.PNPSolve to determin pose of the camera in world coodinates
                camera_3dp_w = cv2.solvePnP(aruco_known_3dp_w,aruco_2dp_image,self.arm_camera_mtx,self.arm_camera_dist)
            
                # Append the "Pose" 3d position of camera in world coordinates in camera_3dp_world
                arm_camera_3dp_world.append()

                # Calculate the "Pose" of the camera in Robot Coordinates
                transform = tf_buffer.lookup_transform(
                    'base_link',        # target frame
                    'camera_link',      # source frame
                    rclpy.time.Time(),  # latest available
                    timeout=rclpy.Duration(seconds=1.0)
                )
                self.get_logger().info(f"Got transform: {transform.transform.translation}")

                t = transform.transform.translation

                arm_camera_3dp_robot.append([t.x,t.y,t.z])
            
                

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

        # Delete ROS Objects
        del tf_listener
        del tf_buffer
