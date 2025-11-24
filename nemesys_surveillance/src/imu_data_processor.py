#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np


class ImuDataProcessorNode:
    def __init__(self):
        rospy.init_node('imu_data_processor')

        self.imu_subscriber = rospy.Subscriber("/raw_imu_data", Imu, self.imu_callback)
        self.angle_axis_pubisher = rospy.Publisher("/euler_angles", Vector3, queue_size = 1)

    def imu_callback(self, data):
        quat = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        if quat is None:
            roll_x = 0
            pitch_y = 0
            yaw_z = 0
        else:
            q = quat/np.sqrt(np.dot(quat, quat))
            x = q[0]
            y = q[1]
            z = q[2]
            w = q[3]

            # Roll (x-axis rotation)
            sin_phi = +2.0 * (w * x + y * z)
            cos_phi = +1.0 - 2.0 * (x * x + y * y)
            roll_x = np.rad2deg(np.arctan2(sin_phi, cos_phi))
            
            # Pitch (y-axis rotation)
            sin_theta = +2.0 * (w * y - z * x)
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            pitch_y = np.rad2deg(np.arcsin(sin_theta))
            
            # Yaw (z-axis rotation)
            sin_psi = +2.0 * (w * z + x * y)
            cos_psi = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = np.rad2deg(np.arctan2(sin_psi, cos_psi))
        
        angles = Vector3()
        angles.x = roll_x
        angles.y = pitch_y
        angles.z = yaw_z
        self.angle_axis_pubisher.publish(angles)
    
if __name__ == '__main__':
    try:
        node = ImuDataProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 