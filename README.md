# Policy Deployment for Go1 Robots
[![Ubuntu 20.04](https://img.shields.io/badge/Ubuntu-20.04-yellow)](https://ubuntu.com/20-04)
[![ROS Noetic](https://img.shields.io/badge/ROS_Noetic-20.04-green)](https://wiki.ros.org/noetic)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://docs.python.org/3/whatsnew/3.10.html)

- This repository provides deployment code for Go1 policies trained in IsaacLab, such as [LocoTouch](https://arxiv.org/abs/2505.23175).
- The codebase is structured to support easy customization and minor modifications for new policies.


### Table of Contents
1. [Installation](#installation)  
2. [Go1 Policy Deployment](#go1-policy-deployment)


## Installation <a name="installation"></a>
#### Basic Installation
1. Create a conda environment with python 3.10:
   ```bash
   cd Go1-Policy-Deployment
   conda create -n policy_deploy python=3.10
   ```
2. Install the dependencies:
   ```bash
   conda activate policy_deploy
   pip install -e .
   ```
3. Install [ROS Neotic](https://wiki.ros.org/noetic/Installation/Ubuntu) (we only test the code on Ubuntu 20.04).

#### Go1 SDK Installation
1. Download the SDK:
   ```bash
   git clone https://github.com/unitreerobotics/unitree_legged_sdk.git
   ```
2. Make sure the required packages are installed, following Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk). Most notably, please make sure to install `Boost` and `LCM`:
   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   pip install empy catkin_pkg
   ```
3. Then, go to the `unitree_legged_sdk` directory and build the libraries:
   ```bash
   cd unitree_legged_sdk
   mkdir build && cd build
   cmake -DPYTHON_BUILD=TRUE ..
   make
   cd ../../
   ```

## Go1 Policy Deployment <a name="running_real"></a>

#### Prerequisites:
1. Launch ROS:
   ```bash
   roscore
   ```
2. Start the joystick interface (see [locomotion_cfg.py](/config/locomotion_cfg.py) for configuration details):
   ```bash
   python teleoperation/joystick.py
   ```

#### Locomotion:
```bash
python deploy/locomotion.py
```

#### State-Based Object Transport (Teacher Policy):
1. Run the OptiTrack motion capture system:
   ```bash
   python mocap/run_optitrack.py
   ```
2. Run the teacher policy:
   ```bash
   python deploy/transport_teacher.py
   ```

#### Tactile-Aware Object Transport (Student Policy):
1. Run the tactile sensor:
   ```bash
   python tactile_sensing/run_tactile_sensing.py
   ```
2. Run the student policy:
   ```bash
   python deploy/transport_student.py
   ```







