Repository: nemesys_surveillance (ROS / Gazebo package)

Purpose
- Help an AI coding agent become productive in this repository: how it is structured, how components connect, and common developer commands.

Quick facts
- Package manifest: `package.xml` declares the package name as `nemesys_controller` (note: the repository folder is `nemesys_surveillance`). Treat the ROS package name as the authoritative name for `roslaunch` / `rosrun`.
- Build system: Catkin (CMake). Many nodes are Python scripts installed with `catkin_install_python` in `CMakeLists.txt`.
- Simulation: Gazebo is used heavily. Key launch files live in `launch/` (e.g. `launch/nemesys_rov.launch`).

Where to look first
- `launch/` — runtime wiring: which Gazebo worlds, model spawns, and nodes are launched together. Example: `launch/nemesys_rov.launch` spawns the URDF, acoustic source, and nodes such as `nemesys_control_node` and `tdoa_estimator`.
- `src/*.py` — main ROS nodes. Examples:
  - `src/nemesys_control_node.py` — subscribes to `/nemesys/user_input` and `/gazebo/link_states`, publishes per-thruster `Wrench` topics (`/front_right_thrust`, etc.).
  - `src/nemesys_teleop_keyboard.py` — interactive keyboard teleop (uses `termios`/TTY; run in a real terminal).
  - `src/tdoa_estimator.py`, `src/localizer*.py` — acoustic localization nodes that integrate with acoustic model spawns in Gazebo.
- `urdf/nemesys.urdf.xacro` — robot model used by the launch files and `robot_state_publisher`.
- `meshes/`, `models/` and `worlds/` — Gazebo assets; launch files reference them via `$(find <package>)/...`.

Important patterns & gotchas
- Package vs directory name: the `package.xml` and `CMakeLists.txt` use the name `nemesys_controller` while the folder is `nemesys_surveillance`. When invoking `roslaunch`/`rosrun`, prefer the declared package name (`nemesys_controller`) or use full path to launch files if the package cannot be resolved.
- Python3 usage: Nodes have `#!/usr/bin/env python3` shebangs. Ensure your ROS distro and environment support Python 3 (Noetic or later) or run the nodes inside a Python3-enabled workspace.
- Executables are installed by `catkin_install_python` in `CMakeLists.txt`; after build the scripts are available via `rosrun nemesys_controller <script>`.
- Topics are mostly absolute (leading `/`). Example topic names to inspect: `/nemesys/user_input`, `/gazebo/link_states`, `/deviation_error`, `/front_right_thrust`.
- Teleop node reads TTY directly — do not run it in an environment without a controlling TTY (CI, some editors). Use a normal terminal.

Common developer commands (assumes workspace root is the catkin workspace)
- Build the workspace (from catkin workspace root):
```
# replace <distro> with your ROS distro (e.g. noetic)
source /opt/ros/<distro>/setup.bash
catkin_make
source devel/setup.bash
```
- Launch the sim (example):
```
roslaunch nemesys_controller nemesys_rov.launch
```
If `roslaunch` cannot find the package, call with full path to the launch file:
```
roslaunch /home/adnana/catkin_ws/src/ALaGatorSim/nemesys_surveillance/launch/nemesys_rov.launch
```
- Run a single node directly with rosrun (after build):
```
rosrun nemesys_controller nemesys_teleop_keyboard.py
```

Debugging & inspection
- Visualize runtime graph: `rosrun rqt_graph rqt_graph`.
- Inspect topics: `rostopic list`, `rostopic echo /deviation_error`.
- Inspect TF frames: `rosrun tf tf_monitor` and `rosrun rqt_tf_tree rqt_tf_tree`.
- View Gazebo logs and stdout by running `roslaunch` with `--screen` or watching the terminal that launched Gazebo (`gzserver`/`gzclient`).

Cross-package integration
- This package expects message types from `nemesys_interfaces` (`msg/nemesysInput.msg`, `msg/nemesysStatus.msg`). Build order matters: ensure `nemesys_interfaces` is built first.
- Launch files spawn Gazebo models from other packages using `$(find <pkg>)/models/...` (e.g. `underwater_acoustics_sim` or `uw_acoustics`). Verify the referenced package names exist and are on `ROS_PACKAGE_PATH`.

When making changes
- If you add or edit Python nodes, ensure:
  - The file is executable (`chmod +x`) and/or installed via `catkin_install_python`.
  - The node's package dependencies appear in `package.xml` and `CMakeLists.txt` (rospy, std_msgs, tf, scipy/numpy if used; system deps for scipy/numpy may be needed).
- For C++ plugin work (Gazebo plugins in `uw_acoustics/src/*.cpp`), follow the patterns in `uw_acoustics/CMakeLists.txt` and rebuild with `catkin_make`.

Examples for common tasks
- Run the full ROV sim (Gazebo + nodes):
```
source /opt/ros/<distro>/setup.bash
cd /home/adnana/catkin_ws
catkin_make
source devel/setup.bash
roslaunch nemesys_controller nemesys_rov.launch
```
- If you need to examine why thruster forces are zero: `rostopic echo /front_right_thrust`, inspect `src/nemesys_control_node.py` (look at `input_callback` and `states_callback`).

If anything is unclear or you want a different focus (tests, CI, contributor guide), tell me which area to expand and I'll iterate.
