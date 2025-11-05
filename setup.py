from setuptools import setup

package_name = 'tile_pick_stack'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/launch', ['launch/tile_pick_stack.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='steve',
    maintainer_email='steve@example.com',
    description='Vision + MoveIt integration for NED-2 pick and stack demo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vision_pick_stack = tile_pick_stack.src.vision_pick_stack:main',
        ],
    },
)
