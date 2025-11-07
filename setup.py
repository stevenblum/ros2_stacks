from setuptools import setup

package_name = 'ros2_stacks'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/launch', ['launch/ros_stacks.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='steven.blum@uri.edu',
    description='Stack Tiles',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'ros_stacks = ros2_stacks.src.ros_stacks:main',
        ],
    },
)
