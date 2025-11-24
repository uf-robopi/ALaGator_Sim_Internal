#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
from gazebo_msgs.msg import LinkStates
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class TrajectoryVisualizer:
    def __init__(self):
        rospy.init_node('trajectory_visualizer')

        self.traj_marker = Marker()
        self.traj_marker.header.frame_id = "map"
        self.traj_marker.type = Marker.LINE_STRIP
        self.traj_marker.action = Marker.ADD
        self.traj_marker.scale.x = 0.05  # line width
        self.traj_marker.color.r = 1.0
        self.traj_marker.color.g = 0.0
        self.traj_marker.color.b = 0.0
        self.traj_marker.color.a = 1.0

        # FIX: Set orientation to identity
        self.traj_marker.pose.orientation.x = 0.0
        self.traj_marker.pose.orientation.y = 0.0
        self.traj_marker.pose.orientation.z = 0.0
        self.traj_marker.pose.orientation.w = 1.0

        self.pub = rospy.Publisher('/trajectory_marker', Marker, queue_size=10)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_callback)

    def link_callback(self, msg):
        try:
            idx = msg.name.index("nemesys::base_link")
            pos = msg.pose[idx].position
            p = Point(x=pos.x, y=pos.y, z=pos.z)
            self.traj_marker.points.append(p)
            self.traj_marker.header.stamp = rospy.Time.now()
            self.pub.publish(self.traj_marker)
        except ValueError:
            pass

if __name__ == '__main__':
    TrajectoryVisualizer()
    rospy.spin()
