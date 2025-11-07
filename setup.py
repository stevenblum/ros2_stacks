import os
from glob import glob
from setuptools import setup, find_packages   # <-- change here

package_name = 'ros2_stacks'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['ros2_stacks', 'ros2_stacks.*']),  # <-- change here
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='steven.blum@uri.edu',
    description='Stack Tiles',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'ros2_stacks = ros2_stacks.ros2_stacks:main',  # <-- correct for flattened structure
        ],
    },
)
