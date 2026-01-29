#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import time
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from nemesys_interfaces.msg import nemesysInput as NemesysInput 

class LawnmowerExecutor:
    def __init__(self):
        rospy.init_node('lawnmower_executor')

        # Internal state
        self.speed_cmps = 40.0
        self.t1 = 5.0  # Duration along X
        self.t2 = 3.0  # Duration along Y
        self.repetitions = 3
        self.target_depth = -1.6
        self.current_yaw = None

        self.max_speed = 0.05  # 5% of max speed
        self.heave = 0.0
        self.cmd = NemesysInput()
        self.cmd.roll = 0.0

        # Motion state
        self.state = None
        self.phase = 0  # 0–7 repeated in cycle
        self.cycle_count = 0
        self.state_start_t = None
        self.target_yaw = None

        # ROS interfaces
        rospy.Subscriber('/nemesys/speed', Float32, self.speed_callback)
        rospy.Subscriber('/nemesys/side_duration', Float32, self.t1_callback)
        rospy.Subscriber('/nemesys/t2_duration', Float32, self.t2_callback)
        rospy.Subscriber('/nemesys/repetitions', Float32, self.reps_callback)
        rospy.Subscriber('/nemesys/target_depth', Float32, self.depth_callback)
        rospy.Subscriber('/nemesys/euler_angles', Vector3, self.euler_callback)
        rospy.Subscriber('/nemesys/heave_control_input', Float32, self.depth_control_callback)

        self.cmd_pub = rospy.Publisher('/nemesys/user_input', NemesysInput, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update)

        rospy.loginfo("Nemesys LawnmowerExecutor initialized")

    def speed_callback(self, msg): self.speed_cmps = msg.data
    def t1_callback(self, msg): self.t1 = msg.data
    def t2_callback(self, msg): self.t2 = msg.data
    def reps_callback(self, msg): self.repetitions = int(msg.data)
    def depth_callback(self, msg): self.target_depth = msg.data - 60
    def euler_callback(self, msg): self.current_yaw = msg.z % 360
    def depth_control_callback(self, msg): self.heave = msg.data * self.max_speed

    def update(self, event):
        if self.current_yaw is None:
            return

        if self.state is None:
            self.state = 'go_to_target_depth'
            rospy.loginfo(f"State set to: {self.state}")
            return

        if self.state == 'go_to_target_depth':
            self.cmd.heave = self.heave
            if abs(self.cmd.heave) < 0.05:
                rospy.loginfo("Target depth reached. Starting lawnmower.")
                self.state = 'execute'
                self.phase = 0
                self.state_start_t = time.time()

        elif self.state == 'execute':
            if self.cycle_count >= self.repetitions:
                rospy.loginfo("Lawnmower pattern complete.")
                self.cmd = NemesysInput()  # Stop
                self.state = 'done'
                return

            self.cmd.heave = 0.0
            now = time.time()
            elapsed = now - self.state_start_t

            if self.phase == 0:  # Forward along X
                self.cmd.surge = min(self.speed_cmps * 0.01, 1.0)
                self.cmd.yaw = 0.0
                if elapsed >= self.t1:
                    self.phase = 1
                    self.target_yaw = (self.current_yaw - 90) % 360
                    self.state_start_t = now
            elif self.phase == 1:  # Rotate 90 CW
                diff = self.shortest_yaw_diff(self.current_yaw, self.target_yaw)
                self.cmd.surge = 0.0
                self.cmd.yaw = 0.3 if diff < 0 else -0.3
                if abs(diff) < 3.0:
                    self.cmd.yaw = 0.0
                    self.phase = 2
                    self.state_start_t = now
            elif self.phase == 2:  # Forward along Y
                self.cmd.surge = min(self.speed_cmps * 0.01, 1.0)
                self.cmd.yaw = 0.0
                if elapsed >= self.t2:
                    self.phase = 3
                    self.target_yaw = (self.current_yaw - 90) % 360
                    self.state_start_t = now
            elif self.phase == 3:  # Rotate 90 CW
                diff = self.shortest_yaw_diff(self.current_yaw, self.target_yaw)
                self.cmd.surge = 0.0
                self.cmd.yaw = 0.3 if diff < 0 else -0.3
                if abs(diff) < 3.0:
                    self.cmd.yaw = 0.0
                    self.phase = 4
                    self.state_start_t = now
            elif self.phase == 4:  # Forward along X (return)
                self.cmd.surge = min(self.speed_cmps * 0.01, 1.0)
                self.cmd.yaw = 0.0
                if elapsed >= self.t1:
                    self.phase = 5
                    self.target_yaw = (self.current_yaw + 90) % 360
                    self.state_start_t = now
            elif self.phase == 5:  # Rotate 90 CCW
                diff = self.shortest_yaw_diff(self.current_yaw, self.target_yaw)
                self.cmd.surge = 0.0
                self.cmd.yaw = 0.3 if diff < 0 else -0.3
                if abs(diff) < 3.0:
                    self.cmd.yaw = 0.0
                    self.phase = 6
                    self.state_start_t = now
            elif self.phase == 6:  # Forward along Y
                self.cmd.surge = min(self.speed_cmps * 0.01, 1.0)
                self.cmd.yaw = 0.0
                if elapsed >= self.t2:
                    self.phase = 7
                    self.target_yaw = (self.current_yaw + 90) % 360
                    self.state_start_t = now
            elif self.phase == 7:  # Rotate 90 CCW
                diff = self.shortest_yaw_diff(self.current_yaw, self.target_yaw)
                self.cmd.surge = 0.0
                self.cmd.yaw = 0.3 if diff < 0 else -0.3
                if abs(diff) < 3.0:
                    self.cmd.yaw = 0.0
                    self.cycle_count += 1
                    self.phase = 0
                    self.state_start_t = now
                    rospy.loginfo(f"Cycle {self.cycle_count} complete.")

        elif self.state == 'done':
            self.cmd.surge = 0.0
            self.cmd.heave = 0.0
            self.cmd.yaw = 0.0

        # Publish command
        self.cmd_pub.publish(self.cmd)

    def shortest_yaw_diff(self, current, target):
        """Returns signed shortest difference from current to target yaw (degrees)."""
        return (target - current + 180) % 360 - 180

if __name__ == '__main__':
    try:
        node = LawnmowerExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
