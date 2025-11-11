#!/usr/bin/env python3
from mcmot import MCMOTracker
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Pose
import tf2_ros
import random
from time import time
from copy import deepcopy
import os
import numpy as np
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
from functools import partial

ARM_CAMERA_IMAGE_TOPIC = '/niryo_robot_vision/compressed_video_stream'
FIXED_CAMERA_LEFT_IMAGE_TOPIC = '/cam_left/my_fixed_camera/image_raw'
FIXED_CAMERA_RIGHT_IMAGE_TOPIC = '/cam_right/my_fixed_camera/image_raw'
STACK_POSE_0 = np.array([0.3,0.0,.005])
GRIP_WIDTH_ON_TILE = .039
APPROACH_HEIGHT = 0.1
SPEED_ACCEL_FACTOR = .03
WAIT_FOR_USER_VALIDATION_BEFORE_MOVES = True
TILE_MIN_DIST_FROM_STACK = .04
TILE_MAX_DIST_FROM_STACK = .3

class TilePublisher(Node):
    def __init__(self):
        super().__init__('tile_publisher')

        self.wait_for_user_validation_before_moves = WAIT_FOR_USER_VALIDATION_BEFORE_MOVES

        # Camera Tracking and Calibration Settings
        self.w2r_calibration_status = False
        self.w2r_calibration_time = None
        self.R_w2r = None
        self.t_w2r = None

        self.arm_camera_mtx = None # 
        self.arm_camera_dist = None #

        self.mcmot_calibration_status = False
        self.mcmot = MCMOTracker("LATEST",[0,1],["lab_logitech1","lab_logitech2"], "4square", display=True)
        self.mcmot_calibration_status = True
        self.mcmot_calibration_time = time()

        self.robot_axis_c= [None for _ in range(len(self.mcmot.cameras))] # Each camera has its own camera coordinates for robot axis

        # Moveit Commander Group Settings
        roscpp_initialize([])
        self.arm_group = MoveGroupCommander('arm')  # name from MoveIt SRDF
        self.manipulator_group = MoveGroupCommander("manipulator")
        self.gripper_group = MoveGroupCommander("gripper")
        self.arm_group.set_max_velocity_scaling_factor(SPEED_ACCEL_FACTOR )
        self.arm_group.set_max_acceleration_scaling_factor(SPEED_ACCEL_FACTOR )
        self.manipulator_group.set_max_velocity_scaling_factor(SPEED_ACCEL_FACTOR )
        self.manipulator_group.set_max_acceleration_scaling_factor(SPEED_ACCEL_FACTOR )
        self.gripper_group.set_max_velocity_scaling_factor(SPEED_ACCEL_FACTOR )
        self.gripper_group.set_max_acceleration_scaling_factor(SPEED_ACCEL_FACTOR )

        # Pick and Stack Settings
        self.tile_min_dist_from_stack = TILE_MIN_DIST_FROM_STACK
        self.tile_max_dist_from_stack = TILE_MAX_DIST_FROM_STACK
        self.stack_pose_0 = STACK_POSE_0
        self.stack_height = 0.02  # meters between tiles
        self.stack_count = 0
        self.status = "Standby"
        self.status_time = time()

        # Define Callbacks with MultiThreading
        self.cb_img = ReentrantCallbackGroup()
        self.cb_timer = ReentrantCallbackGroup()
        self.frame_lock = threading.Lock()

        self.frame = None
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image, ARM_CAMERA_IMAGE_TOPIC, self.image_callback, 3,
            callback_group=self.cb_img
        )
        self.subscription  # prevent unused variable warning

        self.sub_left_camera = self.create_subscription(
            Image,
            FIXED_CAMERA_LEFT_IMAGE_TOPIC,
            partial(self.fixed_camera_callback, camera_number=0),
            3
        )

        self.sub_right_camera = self.create_subscription(
            Image,
            FIXED_CAMERA_RIGHT_IMAGE_TOPIC,
            partial(self.fixed_camera_callback, camera_number=1),
            3
        )

        self.detect_timer_handle= self.create_timer(0.1, self.detect_callback, callback_group=self.cb_timer)

        self.stack_timer_handle = self.create_timer(0.5, self.stack_callback, callback_group=self.cb_timer)

    def stack_callback(self):
        if not self.w2r_calibration_status:
            success = self.calibrate_w2r()
            if success:
                self.w2r_calibration_status = True
                self.w2r_calibration_time = time()
        
        if time()-self.mcmot_calibration_time < 3:
            return None
        
        if self.status != "Standby" or (time()-self.status_time)<2:
            return None

        self.set_status("Find Tiles to Stack")
        self.mcmot.match_global_tracks() # dict, keys (track_id_cam1,track_id_cam2):[x,y,z]
        tile_poses_world_coord = self.mcmot.global_tracks.values() # gets a list of 3d points
        tile_poses_robot_coord = []
        tile_poses_dist_from_stack = []
        for pose_w in tile_poses_world_coord:
            pose_r = self.transform_w2r(np.array(pose_w,dtype=float))
            tile_poses_robot_coord.append(pose_r)
            tile_poses_dist_from_stack.append(np.linalg.norm(pose_r[:2]-self.stack_pose_0[:2])) # Calculated distance in x,y plane
        
        tile_poses_dist_from_stack=np.array(tile_poses_dist_from_stack,dtype=float)

        self.get_logger().info(f"TILE DETECTION DISTANCE FROM STACK: {tile_poses_dist_from_stack}")

        availible_tiles = int(np.sum( np.logical_and(tile_poses_dist_from_stack>self.tile_min_dist_from_stack,tile_poses_dist_from_stack<self.tile_max_dist_from_stack)))

        self.get_logger().info(f"    {availible_tiles} between {self.tile_min_dist_from_stack} and {self.tile_max_dist_from_stack} from stack")

        if availible_tiles == 0:
            self.get_logger().info("    No tiles are the proper distance from stack, no arm movements issued")
            return None

        tile_poses_dist_from_stack[tile_poses_dist_from_stack<self.tile_min_dist_from_stack] = 10000

        min_index = np.argmin(tile_poses_dist_from_stack)
        min_dist = tile_poses_dist_from_stack[min_index]

        if min_dist<self.tile_min_dist_from_stack or min_dist>self.tile_max_dist_from_stack:
            print("ERROR: Chose a tile outside of the acceptable distance range from stack")
            self.get_logger().info("ERROR: Chose a tile outside of the acceptable distance range from stack")
            return None

        closest_tile_pose = tile_poses_robot_coord[min_index]

        self.get_logger().info(f"Selected Tile at (x,y,z): {closest_tile_pose}")

        self.pick_and_stack(closest_tile_pose)

        self.set_status("Standby")

    def detect_callback(self):
        if not self.w2r_calibration_status:
            return None
        
        if not self.mcmot_calibration_status:
            return None

        self.mcmot.update_cameras_tracks() # frames in cameras updated in image_callbacks
        self.mcmot.match_global_tracks()
        self.mcmot.update_displays()

    def fixed_camera_callback(self, msg, camera_number):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.frame_lock: # required for MultiThreading
            self.mcmot.cameras[camera_number].frame = frame

    def pick_and_stack(self, tile_pose):
        self.get_logger().info(f"Received tile pose: {tile_pose}")

        # Base positions as numpy arrays
        pick_pose = np.array(tile_pose, dtype=float)
        stack_pose = np.array(self.stack_pose_0, dtype=float)
        stack_pose[2] = stack_pose[2] + self.stack_height*self.stack_count 

        # --- Pick approach ---
        pick_approach = deepcopy(pick_pose)
        pick_approach[2] += APPROACH_HEIGHT
        self.set_status("Pick Approach")
        self.safe_go(self.arm_group, pose=pick_approach, label="Pick Approach")

        # --- Open gripper before pick ---
        self.set_status("Open Gripper Before Pick")
        g = self.gripper_group.get_current_joint_values()
        g[0] = GRIP_WIDTH_ON_TILE + 0.03
        self.safe_go(self.gripper_group, joint_goal=g, label="Open Gripper")

        # --- Pick pose ---
        self.set_status("Pick Pose")
        plan1 = self.safe_go(self.arm_group, pose=pick_pose, label="Pick Pose")

        # --- Close gripper to pick ---
        self.set_status("Close Gripper to Pick")
        g = self.gripper_group.get_current_joint_values()
        g[0] = GRIP_WIDTH_ON_TILE
        self.safe_go(self.gripper_group, joint_goal=g, label="Close Gripper")

        # --- Stack approach ---
        stack_approach = deepcopy(stack_pose)
        stack_approach[2] += APPROACH_HEIGHT
        self.set_status("Stack Approach")
        self.safe_go(self.arm_group, pose=stack_approach, label="Stack Approach")

        # --- Stack pose ---
        self.set_status("Stack Pose")
        plan2 = self.safe_go(self.arm_group, pose=stack_pose, label="Stack Pose")

        # --- Open gripper to release ---
        self.set_status("Open Gripper for Stack")
        g = self.gripper_group.get_current_joint_values()
        g[0] = GRIP_WIDTH_ON_TILE + 0.01
        self.safe_go(self.gripper_group, joint_goal=g, label="Open Gripper")

        # --- Stack retreat ---
        stack_retreat = deepcopy(stack_pose)
        stack_retreat[2] += APPROACH_HEIGHT
        self.set_status("Stack Retreat")
        self.safe_go(self.arm_group, pose=stack_retreat, label="Stack Retreat")

        # --- Complete ---
        if plan1 and plan2:
            self.stack_count += 1
            self.get_logger().info(f"STACKED TILE #{self.stack_count}")

    def safe_go(self, group, pose=None, joint_goal=None, label=""):
        if pose is not None:
            if isinstance(pose, (list, tuple, np.ndarray)):
                p = Pose()
                p.position.x, p.position.y, p.position.z = map(float,pose)
                p.orientation.w = 1.0
                group.set_pose_target(p)
            else:
                group.set_pose_target(pose)

        elif joint_goal is not None:
            group.set_joint_value_target(joint_goal)
        else:
            return False

        plan = group.plan()
        success = True
        exec_plan = plan
        if isinstance(plan, tuple):
            success = bool(plan[0])        # check first element of tuple
            exec_plan = plan[1]            # second element is trajectory

        if not success or exec_plan is None:
            self.get_logger().warning(f"Planning failed for '{label}' on '{group.get_name()}'.")
            group.clear_pose_targets()
            return False

        if self.wait_for_user_validation_before_moves:
            if input(f"Execute '{label}' for '{group.get_name()}'? (y/n): ").lower() != 'y':
                group.clear_pose_targets()
                return False

        ok = group.execute(exec_plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        return ok

    def _estimate_rigid_transform(self,P_world: np.ndarray, Q_robot: np.ndarray):
        # P,Q: Nx3
        mu_P = P_world.mean(axis=0)
        mu_Q = Q_robot.mean(axis=0)
        X = P_world - mu_P
        Y = Q_robot - mu_Q
        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_Q - R @ mu_P
        return R, t

    def _camera_center_in_world_from_pnp(self,rvec, tvec):
        # PnP gives: x_cam = R * x_world + t
        R, _ = cv2.Rodrigues(rvec)
        C_w = -R.T @ tvec.reshape(3)
        return C_w

    # ---- inside your class ----

    def transform_w2r(self, world_3dp: np.ndarray) -> np.ndarray:
        world_3dp = np.atleast_2d(world_3dp) # even if only one point is passed, converts to 2d array
        robot_3dp = (self.R_w2r @ world_3dp.T).T + self.t_w2r
        return robot_3dp.squeeze()
    
    def transform_r2w(self, robot_3dp: np.ndarray) -> np.ndarray:
        robot_3dp = np.atleast_2d(robot_3dp) # even if only one point is passed, converts to 2d array
        world_3dp = (self.R_w2r.T @ (robot_3dp - self.t_w2r).T).T
        return world_3dp.squeeze()


    def calibrate_w2r(self):
        arm_cam_3dp_robot = []
        arm_cam_3dp_world = []

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        aruco_config = MCMOTracker.ArucoConfig("square4")

        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, self)

        self.get_logger().info("Calibration: press 'c' to capture, 'q' to finish.")
        while rclpy.ok():
            with self.frame_lock:
                if self.frame is None:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    continue
                frame = self.frame.copy()

            disp = frame
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)
            cv2.putText(disp, "Calibrating World to Robot Transform: Press C to Capture or Q to Close.", (30,30), cv2.FONT_HERSHEY_SIMPLEX, .7,(0,225,0),2)
            cv2.imshow("Arm Camera", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key != ord('c'):
                rclpy.spin_once(self, timeout_sec=0.01)
                continue

            if ids is None or len(ids) < 4:
                self.get_logger().warning("Need ≥4 ArUco markers for a robust PnP capture.")
                continue

            image_pts = []
            world_pts = []
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id not in aruco_config.points:
                    self.get_logger().warning(f"Marker {marker_id} not in config; skipping.")
                    continue
                # use the marker center as the 2D point
                pts = corners[i].reshape(-1, 2)         # 4x2
                center_2d = pts.mean(axis=0)            # 2,
                image_pts.append(center_2d)
                world_pts.append(aruco_config.points[marker_id])  # 3,

            if len(world_pts) < 4:
                self.get_logger().warning("After filtering, <4 usable markers. Try again.")
                continue

            image_pts = np.asarray(image_pts, dtype=np.float32)
            world_pts = np.asarray(world_pts, dtype=np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                world_pts, image_pts,
                self.arm_camera_mtx, self.arm_camera_dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                self.get_logger().warning("solvePnP failed; try again.")
                continue

            cam_pos_world = self._camera_center_in_world_from_pnp(rvec, tvec)

            try:
                rclpy.spin_once(self, timeout_sec=0.05)
                tf = tf_buffer.lookup_transform(
                    'base_link', 'camera_link',
                    rclpy.time.Time(), timeout=Duration(seconds=3.0)
                )
                t = tf.transform.translation
                cam_pos_robot = np.array([t.x, t.y, t.z], dtype=float)
            except Exception as e:
                self.get_logger().warning(f"TF lookup failed: {e}")
                continue

            arm_cam_3dp_world.append(cam_pos_world)
            arm_cam_3dp_robot.append(cam_pos_robot)
            self.get_logger().info(f"Captured pair #{len(arm_cam_3dp_world)}")

        cv2.destroyWindow("Arm Camera")

        if len(arm_cam_3dp_world) < 3:
            self.get_logger().error("Not enough captures to estimate transform (need ≥3).")
            del tf_listener; del tf_buffer
            return False

        Pw = np.asarray(arm_cam_3dp_world, dtype=float)   # Nx3
        Qr = np.asarray(arm_cam_3dp_robot, dtype=float)   # Nx3
        R, t = self._estimate_rigid_transform(Pw, Qr)

        self.R_w2r = R
        self.t_w2r = t
        self.get_logger().info(f"R_w2r:\n{R}\nt_w2r: {t}")
        del tf_listener; del tf_buffer
        return True

    def draw_robot_axis(self,frame,camera_number):
        if self.robot_axis_c[camera_number] == None:
            axis_length = 10
            robot_axis_points_r = np.float32([[0,0,0],[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
            robot_axis_points_w = self.transform_r2w(robot_axis_points_r)
            robot_axis_points_c = self.mcmot.cameras[0].transform_w2c(robot_axis_points_w)
            self.robot_axis_c[camera_number] = []
            for i in range(4):
                self.robot_axis_c[camera_number].append(tuple(int(x) for x in robot_axis_points_c [i].ravel()))

        frame = cv2.line(frame, self.robot_axis_c[camera_number][0], self.robot_axis_c[camera_number][1], (255,0,0), 3)  # X-axis in blue
        frame = cv2.line(frame, self.robot_axis_c[camera_number][0], self.robot_axis_c[camera_number][2], (0,255,0), 3)  # Y-axis in green
        frame = cv2.line(frame, self.robot_axis_c[camera_number][0], self.robot_axis_c[camera_number][3], (0,0,255), 3)  # Z-axis in red

        return frame

    def set_status(self,status):
        self.status = status
        self.status_time = time()
        self.get_logger().info(f"Node Status Set to: {status},{self.status_time}")


def main(args=None):
    rclpy.init(args=args)
    node = TilePublisher()
    executor = MultiThreadedExecutor(num_threads=4)  # at least 2; 3–4 is nice
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
