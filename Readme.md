# **Underwater Data Center (UDC) Surveillance with NemeSys AUV**

This repository provides a complete **ROS1 Noetic + Gazebo 11** simulation environment for UDC surveillance. Particularly, it simulates a pipeline for underwater adversarial acoustic source localization.

The system simulates:

- An **underwater world** with cylindrical UDC pods and seabed.   
- The NemeSys **surveillance AUV** with 4 DOF controller and waypoint-based mission executor.  
- A custom **acoustic source** and **hydrophones** for underwater acoustic attack simulation and Time/Frequency-Difference-of-Arrival (TDOA/FDOA)-based localization.  
- Gazebo world and RViz visualization including trajectories, acoustic markers, pod meshes, etc.  



## **Repository Structure**

The repository contains three ROS packages:

### **1. `nemesys_surveillance`**
Core agentic surveillance package providing:
- Launch files  
- Gazebo world setup  
- Pod renderer  
- AUV motion controller  
- 3D waypoint-based mission executor  
- RViz visualizers 
- Large assets in:
  - `models/`
  - `meshes/`
  - `Media/` (materials & textures)

### **2. `uw_acoustics`**
Acoustic simulation & localization package:
- Acoustic source model
- Static and dynamic hydrophone model  
- TDOA estimator  
- FDOA estimator   

### **3. `nemesys_interfaces`**
Defines custom ROS message types for 4-DOF control of nemesys.

---

## **Large Files (Meshes, Models, Materials)**

Large assets (meshes, worlds, materials, etc.) are stored externally.

**Download the full asset bundle here:**  
[https://www.dropbox.com/scl/fo/oyp253iitg4wk3duw7fss/ACCR62KCgdnW99tOCrzQYTQ?rlkey=kyc3nuu7rwupilwtk0dgse9yl&st=n3i0ucwc&dl=0](https://www.dropbox.com/scl/fo/oyp253iitg4wk3duw7fss/ACCR62KCgdnW99tOCrzQYTQ?rlkey=kyc3nuu7rwupilwtk0dgse9yl&st=n3i0ucwc&dl=0)

After downloading, place the contents inside the root directory of respective pacakge.


---

## **Requirements**

- **ROS Noetic**
- **Gazebo 11 (Classic)**
- **Ubuntu 20.04**
- `gazebo_ros` + standard ROS desktop packages
- Python 3

---

## **Setup**

Clone the repository into a catkin workspace and build:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone <this_repo_url>
cd ..
catkin_make
source devel/setup.bash
```

## **How to Run**

Launch the localizer pipeline with full underwater world + surveillance AUV + adversarial agent.

``` bash
roslaunch nemesys_surveillance localizer.launch
