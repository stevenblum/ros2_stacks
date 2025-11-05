#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown

STACK_POSE_0 = [0.3,0.0,.005]

class PickAndStack(Node):
    def __init__(self):
        super().__init__('pick_and_stack')
        roscpp_initialize([])
        self.group = MoveGroupCommander('arm')  # name from MoveIt SRDF
        self.subscription = self.create_subscription(PoseStamped, '/tile_pose', self.callback, 10)
        self.stack_height = 0.02  # meters between tiles
        self.stack_count = 0

    def callback(self, pose):
        self.get_logger().info(f"Received tile pose: {pose.pose.position}")
        pick_pose = pose.pose
        stack_pose = PoseStamped()
        stack_pose.header.frame_id = pose.header.frame_id
        stack_pose.pose.position.x = STACK_POSE_0[0]
        stack_pose.pose.position.y = STACK_POSE_0[1]
        stack_pose.pose.position.z = STACK_POSE_0[2] + self.stack_height * self.stack_count
        stack_pose.pose.orientation.w = 1.0

        # Move to pick
        self.group.set_pose_target(pick_pose)
        plan1 = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        # Move to stack
        self.group.set_pose_target(stack_pose.pose)
        plan2 = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if plan1 and plan2:
            self.stack_count += 1
            self.get_logger().info(f"Stacked tile #{self.stack_count}")

def main(args=None):
    rclpy.init(args=args)
    node = PickAndStack()
    rclpy.spin(node)
    roscpp_shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
