#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import os
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

def make_marker(mesh_path, frame_id, ns, mid, pose, color=(0.8,0.8,0.8,1.0), scale=(1.0,1.0,1.0)):
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = mid
    m.type = Marker.MESH_RESOURCE
    m.action = Marker.ADD
    m.pose = pose
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.mesh_resource = "file://" + mesh_path
    m.frame_locked = False
    m.lifetime = rospy.Duration(0)  # forever
    return m

def pose_xyz_rpy(x, y, z, roll, pitch, yaw):
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    q = quaternion_from_euler(roll, pitch, yaw)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p

def main():
    rospy.init_node("pod_array_mesh_publisher")

    # Params (edit path or set via rosparam)
    mesh_path = rospy.get_param("~mesh_path", "/home/adnana/catkin_ws/src/nemesys_surveillance_pkgs/nemesys_description/meshes/pod_pipeline_merged.stl")
    frame_id  = rospy.get_param("~frame_id", "world")
    topic     = rospy.get_param("~topic", "/pod_mesh")

    pub = rospy.Publisher(topic, Marker, queue_size=10, latch=True)

    # Poses copied from the world file (x y z roll pitch yaw)
    pods = [
        ("pod_origin",          (  0.0,   0.0, 0.0, 0.0, 0.0,  0.0      )),
        ("pod_pos30_rotated",   ( -1.38,  5.15, 0.0, 0.0, 0.0,  0.5235  )),
        ("pod_neg30_rotated",   ( -1.38, -5.15, 0.0, 0.0, 0.0, -0.5235  )),
        ("pod_pos90_rotated",   (-10.30, 10.30, 0.0, 0.0, 0.0,  1.5708  )),
        ("pod_neg90_rotated",   (-10.30,-10.30, 0.0, 0.0, 0.0, -1.5708  )),
    ]

    markers = []
    for idx, (name, (x,y,z, r,p,yaw)) in enumerate(pods):
        pose = pose_xyz_rpy(x, y, z, r, p, yaw)
        m = make_marker(mesh_path, frame_id, ns="pod_mesh", mid=idx, pose=pose)
        markers.append(m)

    rate = rospy.Rate(1)  # publish/update once per second
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        for m in markers:
            m.header.stamp = now
            pub.publish(m)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
