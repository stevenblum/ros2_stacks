# stackws

This repository was created from the existing VS Code workspace at /home/scblum/Projects/cleanup_pkg and will be published on GitHub as `stackws`.

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

