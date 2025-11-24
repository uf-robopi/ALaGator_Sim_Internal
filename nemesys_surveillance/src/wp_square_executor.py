#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import math
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import LinkStates
from nemesys_interfaces.msg import nemesysInput as NemesysInput

class SquareExecutor:
    def __init__(self):
        rospy.init_node('square_wp_executor')

        # === Parameters ===
        self.init_x = rospy.get_param("~initial_x", -4.0)
        self.init_y = rospy.get_param("~initial_y", -3.0)
        self.init_yaw_deg = rospy.get_param("~initial_yaw", 0.0)
        self.side_length = rospy.get_param("~side_length", 5.5)
        self.dist_tol = rospy.get_param("~distance_tolerance", 0.1)
        self.yaw_tol = rospy.get_param("~yaw_tolerance", 1.0)  # degrees
        self.speed = rospy.get_param("~speed", 0.6)  # 0-1 (fraction of max)

        # === Internal state ===
        self.current_pos = None
        self.current_yaw = None
        self.cmd = NemesysInput()
        self.cmd.roll = 0.0
        self.cmd.heave = 0.0

        self.state = "go_to_start"
        self.current_wp_idx = 0
        self.waypoints = []
        self.target_yaw = None

        # === Subscribers ===
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_callback)
        rospy.Subscriber('/euler_angles', Vector3, self.euler_callback)
        rospy.Subscriber('/heave_control_input', Float32, self.depth_control_callback)

        # === Publisher ===
        self.cmd_pub = rospy.Publisher('/nemesys/user_input', NemesysInput, queue_size=1)

        # Timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.update)

        rospy.loginfo("Nemesys SquareExecutor (waypoint-based) initialized")

    # -----------------------------
    def euler_callback(self, msg):
        self.current_yaw = msg.z % 360

    def depth_control_callback(self, msg):
        # Just pass-through
        self.cmd.heave = msg.data * 0.05

    def link_callback(self, msg):
        try:
            idx = msg.name.index("nemesys::base_link")
            pose = msg.pose[idx]
            self.current_pos = np.array([pose.position.x, pose.position.y])
        except ValueError:
            pass

    # -----------------------------
    def update(self, event):
        if self.current_pos is None or self.current_yaw is None:
            print("Waiting for initial position and orientation...")
            return

        if self.state == "go_to_start":
            rospy.loginfo("Going to initial waypoint...")
            # Drive to initial (x, y)
            target = np.array([self.init_x, self.init_y])
            dist = np.linalg.norm(self.current_pos - target)
            target_yaw = math.degrees(math.atan2(target[1]-self.current_pos[1],
                                                 target[0]-self.current_pos[0]))

            yaw_error = self.shortest_yaw_diff(self.current_yaw, target_yaw)
            

            if dist > self.dist_tol:
                if abs(yaw_error) > self.yaw_tol:
                    self.cmd.surge = 0.0
                    self.cmd.yaw = 0.3 if yaw_error < 0 else -0.3
                    print(f"Yaw error: {yaw_error:.2f} degrees, adjusting yaw")
                else:
                    self.cmd.surge = self.speed
                    self.cmd.yaw = 0.0
            else:
                rospy.loginfo("Reached initial waypoint. Orienting to initial yaw.")
                self.state = "orient_initial"
                self.target_yaw = self.init_yaw_deg

        elif self.state == "orient_initial":
            yaw_error = self.shortest_yaw_diff(self.current_yaw, self.target_yaw)
            if abs(yaw_error) > self.yaw_tol:
                self.cmd.surge = 0.0
                self.cmd.yaw = 0.3 if yaw_error < 0 else -0.3
            else:
                rospy.loginfo("Initial yaw achieved. Generating square waypoints.")
                self.compute_square_waypoints()
                self.current_wp_idx = 0
                self.state = "goto_waypoint"

        elif self.state == "goto_waypoint":
            if self.current_wp_idx >= len(self.waypoints):
                rospy.loginfo("Completed all waypoints.")
                self.cmd = NemesysInput()
                self.state = "done"
                return

            target = self.waypoints[self.current_wp_idx]
            dist = np.linalg.norm(self.current_pos - target)
            target_yaw = math.degrees(math.atan2(target[1]-self.current_pos[1],
                                                 target[0]-self.current_pos[0]))
            yaw_error = self.shortest_yaw_diff(self.current_yaw, target_yaw)

            if dist > self.dist_tol:
                if abs(yaw_error) > self.yaw_tol:
                    self.cmd.surge = 0.0
                    self.cmd.yaw = 0.3 if yaw_error < 0 else -0.3
                else:
                    self.cmd.surge = self.speed
                    self.cmd.yaw = 0.0
            else:
                rospy.loginfo(f"Waypoint {self.current_wp_idx+1} reached.")
                self.current_wp_idx += 1

        elif self.state == "done":
            self.cmd.surge = 0.0
            self.cmd.yaw = 0.0
            self.cmd.heave = 0.0

        # Publish command
        self.cmd_pub.publish(self.cmd)

    # -----------------------------
    def compute_square_waypoints(self):
        # Convert initial yaw to radians
        yaw_rad = math.radians(self.init_yaw_deg)

        # Corner 1 (relative to init)
        p1 = np.array([self.init_x + self.side_length * math.cos(yaw_rad),
                       self.init_y + self.side_length * math.sin(yaw_rad)])
        # Corner 2
        p2 = np.array([p1[0] - self.side_length * math.sin(yaw_rad),
                       p1[1] + self.side_length * math.cos(yaw_rad)])
        # Corner 3
        p3 = np.array([p2[0] - self.side_length * math.cos(yaw_rad),
                       p2[1] - self.side_length * math.sin(yaw_rad)])
        # Corner 4
        p4 = np.array([p3[0] + self.side_length * math.sin(yaw_rad),
                       p3[1] - self.side_length * math.cos(yaw_rad)])

        self.waypoints = [p1, p2, p3, p4]
        rospy.loginfo(f"Waypoints: {self.waypoints}")

    # -----------------------------
    def shortest_yaw_diff(self, current, target):
        diff = (target - current + 180) % 360 - 180
        if abs(abs(diff) - 180.0) < 5.0:  # within 1 degree of 180
                diff = 180.0  # force consistent direction (always left turn)
        return diff

# ==========================
if __name__ == '__main__':
    try:
        SquareExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
