from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    ros2_stacks_node = Node(
        package='ros2_stacks',
        executable='ros2_stacks',   # 👈 entry point name from setup.py
        name='ros2_stacks_node',
        output='screen',
        emulate_tty=True,  # keeps live terminal color logs
        parameters=[{
            'use_sim_time': False,  # set to True if using Gazebo
        }],
    )

    return LaunchDescription([
        # moveit_launch,  # uncomment if you want MoveIt to launch automatically
        ros2_stacks_node
    ])
