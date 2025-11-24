#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import csv
import os
import numpy as np
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import LinkStates
from datetime import datetime

class FiducialPoseComparator:
    def __init__(self):
        rospy.init_node("fiducial_pose_comparator")

        # Output CSV file
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.expanduser(f"~/pose_comparison_{datetime_str}.csv")
        self.csv_writer = None
        self.file = open(self.csv_file, "w", newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(["timestamp", 
                                  "true_x", "true_y", "true_z", 
                                  "fiducial_x", "fiducial_y", "fiducial_z"])

        self.latest_truth = None
        self.latest_fiducial = None

        rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_callback)
        rospy.Subscriber("/robot_visual_pose", PoseStamped, self.fiducial_callback)

        rospy.loginfo("FiducialPoseComparator node initialized.")

    def link_callback(self, msg):
        try:
            index = msg.name.index("nemesys::base_link")
            pos = msg.pose[index].position
            self.latest_truth = (rospy.Time.now().to_sec(), pos.x, pos.y, pos.z)
        except ValueError:
            pass

    def fiducial_callback(self, msg):
        self.latest_fiducial = (msg.header.stamp.to_sec(),
                                msg.pose.position.x,
                                msg.pose.position.y,
                                msg.pose.position.z)

        # If we also have recent truth, write to file
        if self.latest_truth:
            t_truth, x_t, y_t, z_t = self.latest_truth
            t_fid, x_f, y_f, z_f = self.latest_fiducial

            # Use fiducial timestamp for logging
            self.csv_writer.writerow([t_fid, x_t, y_t, z_t, x_f, y_f, z_f])
            self.file.flush()
            rospy.loginfo_throttle(5.0, f"Logged pose comparison at t={t_fid:.2f}s")

    def __del__(self):
        if self.file:
            self.file.close()

if __name__ == "__main__":
    try:
        FiducialPoseComparator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
