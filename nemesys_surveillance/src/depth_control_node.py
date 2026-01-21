#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================

 
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool
import time, math
import numpy as np
 
 
class DepthControlNode:
    def __init__(self):
        rospy.init_node('depth_control_node')
 
        # Subscribers
        self.depth_sub = rospy.Subscriber(
            name="/raw_depth_data", data_class=LaserScan, callback=self.depth_callback)
        self.depth_sub = rospy.Subscriber(
            name="/target_depth", data_class=Float32, callback=self.target_depth_callback)
        self.depth_change_indicator_sub = rospy.Subscriber(
            name='/depth_change_indicator', data_class=Bool, callback=self.adjust_target_depth_callback)
       
        # Publisher
        self.depth_control_pub = rospy.Publisher(
            name="/heave_control_input", data_class=Float32, queue_size=1)
        
 
        # PID Controller Parameters
        self.error_array = []
        self.dt_array = []
        self.previous_error = 0.0
        self.pid_last_time = None
        self.error_integral = 0.0
        self.count_integrated_errors = 5
        # self.target_depth = 1.4 # meters
        self.target_depth = 58.5 # meters, depth from surface
        self.is_target_depth_changed = False
        self.kp_depth = 1.6
        self.ki_depth = 0.0
        self.kd_depth = 0.9

        # self.depth_log_file = open("/home/alankrit/depth_data_log.txt", "a")
        # self.start_time = time.time()
        # rospy.on_shutdown(self.depth_log_file.close)
 
    def target_depth_callback(self, msg):
        self.target_depth = msg.data #- 60

    def depth_callback(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)

        # Valid beams only
        valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < msg.range_max)
        if not np.any(valid):
            return

        idxs = np.nonzero(valid)[0]
        r_valid = ranges[valid]

        # pick the smallest slant range (closest hit)
        k = int(idxs[np.argmin(r_valid)])
        r_min = float(ranges[k])

        # beam angle for that index
        angle_k = msg.angle_min + k * msg.angle_increment

        # If the selected ray is too oblique, it's not reliable as "depth"
        # (prevents weird behavior at large roll/pitch)
        if abs(angle_k) > math.radians(25):
            return

        # Convert slant range to approximate vertical distance to plane
        depth = r_min * math.cos(angle_k)

        if not np.isfinite(depth) or depth <= 0.0:
            return

        # Optional: reject sudden jumps
        if hasattr(self, "last_depth"):
            if abs(depth - self.last_depth) > 10.0:
                return
        self.last_depth = depth

        if self.is_target_depth_changed:
            self.target_depth = depth
            self.is_target_depth_changed = False

        depth_error = depth - self.target_depth
        heave = self.compute_pid_control(depth_error, self.kp_depth, self.ki_depth, self.kd_depth)

        out = Float32()
        out.data = float(heave)
        self.depth_control_pub.publish(out)

        # log_time = time.time() - self.start_time  # time in seconds since node start
        # self.depth_log_file.write(f"{log_time}, {current_depth}\n")
        # self.depth_log_file.flush()

    def adjust_target_depth_callback(self, msg):
        self.is_target_depth_changed = msg.data

 
 
    # Function to Calculate PID Control
    def compute_pid_control(self, error, kp, ki, kd):
        current_time = time.time()
        if self.pid_last_time is None:
            dt = 0.1 # seconds
        else:
            dt = current_time - self.pid_last_time
        self.pid_last_time = current_time
       
        self.dt_array.append(dt)
        self.error_array.append(error)
        if len(self.error_array) > self.count_integrated_errors:
            self.error_array.pop(0)
            self.dt_array.pop(0)
        self.error_integral = np.dot(self.error_array, self.dt_array)
        error_derivative = (error - self.previous_error) / dt
        control_signal = (kp * error) + (ki * self.error_integral) + (kd * error_derivative)
        control_signal = np.clip(control_signal, -1.0, 1.0)
        self.previous_error = error
 
        return control_signal
 
 
if __name__ == '__main__':
    try:
        node = DepthControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass