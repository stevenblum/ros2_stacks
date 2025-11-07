import os
from glob import glob
from setuptools import setup

package_name = 'ros2_stacks'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # The correct format using os.path.join and glob
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']), # It's good practice to also install the package.xml
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='steven.blum@uri.edu',
    description='Stack Tiles',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'ros2_stacks = ros2_stacks.src.ros2_stacks:main',
        ],
    },
)
