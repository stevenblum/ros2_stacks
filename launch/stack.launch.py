from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Optional: include MoveIt2 bringup if not already running
    # moveit_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         os.path.join(
    #             get_package_share_directory('moveit2_bringup'),
    #             'launch',
    #             'move_group.launch.py'
    #         )
    #     ])
    # )

    vision_pick_stack_node = Node(
        package='tile_pick_stack',
        executable='vision_pick_stack',   # 👈 entry point name from setup.py
        name='vision_pick_stack',
        output='screen',
        emulate_tty=True,  # keeps live terminal color logs
        parameters=[{
            'use_sim_time': False,  # set to True if using Gazebo
        }],
    )

    return LaunchDescription([
        # moveit_launch,  # uncomment if you want MoveIt to launch automatically
        vision_pick_stack_node
    ])
