#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import cv2
import cv2.aruco as aruco
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import math
from tf.transformations import euler_from_matrix

class FiducialPoseEstimator:
    def __init__(self):
        rospy.init_node('fiducial_pose_estimator')

        # Parameters
        self.marker_length = 0.14  # meters
        self.camera_matrix = np.array([[381.36246688113556, 0, 320.5],
                                       [0, 381.36246688113556, 240.5],
                                       [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1))  # assume no distortion

        # Known global poses of each Tag (from SDF)
        self.marker_poses = {
        1: self.make_transform([-0.26, 0, -59.4], [0, -1.57, 0]),  # pod's z is -60 # changed to -1.57 here
        2: self.make_transform([1.19, 0, -59.4], [0, -1.57, 3.14]),
        3: self.make_transform([0.35, 0, -59.07], [0, 0, 0]),
        4: self.make_transform([0.35, -0.41, -59.4], [0, 1.57, -1.57]), #changed to -1.57
        5: self.make_transform([0.35, 0.41, -59.4], [0, -1.57, -1.57]),
}

        # Fixed transform from camera to robot base_link
        self.T_robot_cam = self.make_transform([0., 0., 0.], [math.pi, 0.0, 0.0])

        # ROS setup
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/front_camera/image_raw', Image, self.image_callback)
        self.pose_pub = rospy.Publisher('/robot_visual_pose', PoseStamped, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters =  cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        # markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(gray)

        # self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        # self.parameters = aruco.DetectorParameters_create()

    def make_transform(self, translation, rpy):
        """Create a 4x4 homogeneous transform from translation and RPY"""
        t = tf.transformations.translation_matrix(translation)
        r = tf.transformations.euler_matrix(*rpy)
        return np.dot(t, r)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.marker_poses:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        [corners[i]], self.marker_length, self.camera_matrix, self.dist_coeffs)

                    # Compute transform from marker to camera
                    T_cam_marker = np.eye(4)
                    T_cam_marker[:3, :3] = cv2.Rodrigues(rvec[0][0])[0]
                    T_cam_marker[:3, 3] = tvec[0][0]

                    # Compute global pose
                    T_marker_world = self.marker_poses[marker_id]
                    T_marker_cam = np.linalg.inv(T_cam_marker)
                    T_cam_world = T_marker_world @ T_marker_cam
                    # T_robot_world = T_cam_world @ np.linalg.inv(self.T_robot_cam)
                    T_robot_world = T_marker_world @ np.linalg.inv(T_marker_cam) @ self.T_robot_cam
                    T_robot_world = T_marker_world @ T_marker_cam

                    # Extract and publish robot pose
                    position = tf.transformations.translation_from_matrix(T_robot_world)
                    quat = tf.transformations.quaternion_from_matrix(T_robot_world)

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = 'world'
                    pose_msg.pose.position.x = position[0]
                    pose_msg.pose.position.y = position[1]
                    pose_msg.pose.position.z = position[2]
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]
                    self.pose_pub.publish(pose_msg)

                    # Convert to roll-pitch-yaw (in degrees)
                    roll, pitch, yaw = euler_from_matrix(T_robot_world)
                    roll_deg = math.degrees(roll)
                    pitch_deg = math.degrees(pitch)
                    yaw_deg = math.degrees(yaw)

                    # Print pose nicely
                    print(f"[Pose @ {pose_msg.header.stamp.to_sec():.2f}s] Position: "
                        f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), "
                        f"Orientation (RPY°): ({roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°)")

                    # Stop after the first known marker
                    return


if __name__ == '__main__':
    try:
        FiducialPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
