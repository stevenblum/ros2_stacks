# ros_stack

## TODO for Coding
arm_camera calibration, and loading mtx and dist
URDF for gripper, open/close commands
Calibrate using all 4 corners of aruco marker
For gripper open/close, done use gripper_joint_goal[0], use named joints
Does group commander .go() return a tuple or a trajectory? Depends on ROS/MoveIt version

## TODO Setup/Config
Check that the camera is mounted at the origin of the camera frame
Check the name of the camera frame
Actually calibrate the camera

## TODO for Ros2 Package
How to include python requirments.txt
Node Name

## TODO Function Improvements
Use CV to identify which tile to stack on
... lots more

# File Structure

Note, this repo is a ROS2 package that sits under a workspace and src folder

workspace_folder/
├── src/
│   └── ros2_stacks/ ***THIS REPO***
│       ├── launch/
│       │   └── ros2_stacks.launch.py
│       ├── resource/
│       │   └── ros2_stacks
│       ├── ros2_stacks/
│       │   ├── __init__.py
│       │   └── src/
│       │       └── ros2_stacks.py  # Contains your main() function
│       ├── package.xml
│       ├── setup.cfg
│       └── setup.py
├── build/
├── install/
└── log/


Contents:
- `cleanup_pkg/` — Python package source
- `launch/` — ROS-launch related scripts
- `src/` — helper scripts (e.g. `vision_pick_stack.py`)
- `archive/` — archived/old scripts

How this was set up:
1. Initialize a local git repository
2. Add project files and make an initial commit
3. Create a GitHub repository named `stackws` (via GitHub CLI) and push

If the automated GH push failed, run one of the following locally:

# Using GitHub CLI (recommended if installed and authenticated):
# gh repo create stackws --public --source=. --remote=origin --push

# Or manually (replace USERNAME with your GitHub username):
# git remote add origin https://github.com/USERNAME/stackws.git
# git branch -M main
# git push -u origin main

