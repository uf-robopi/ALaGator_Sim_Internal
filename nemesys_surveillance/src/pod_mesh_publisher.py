#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
from visualization_msgs.msg import Marker
import os

def publish_mesh():
    rospy.init_node('pod_mesh_publisher', anonymous=True)
    pub = rospy.Publisher('/pod_mesh', Marker, queue_size=1)

    marker = Marker()
    marker.header.frame_id = "world"  
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Adjust scale if necessary (1.0 means original size)
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    # Color (RGBA)
    marker.color.r = 0.8
    marker.color.g = 0.8
    marker.color.b = 0.8
    marker.color.a = 1.0

    # Provide full file path
    # mesh_path = "package://nemesys_surveillance/meshes/pod_pipeline_merged.stl" #os.path.abspath("pod_pipeline_merged.stl")
    marker.mesh_resource = marker.mesh_resource = "package://nemesys_surveillance/meshes/pod_pipeline_merged.stl" #"file://" + mesh_path

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_mesh()
    except rospy.ROSInterruptException:
        pass
