#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import math
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Point
from gazebo_msgs.msg import ModelStates

def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

class AcousticSourceMarkers:
    def __init__(self):
        rospy.init_node("acoustic_source_markers")

        # ---- Params ----------------------------------------------------------
        self.frame_id       = rospy.get_param("~frame_id", "world")
        self.model_name     = rospy.get_param("~model_name", "mobile_acoustic_source")

        # initial fallback (used only until /gazebo/model_states arrives)
        self.init_x         = rospy.get_param("~source_x", 0.6)
        self.init_y         = rospy.get_param("~source_y", -2.8)
        self.init_z         = rospy.get_param("~source_z", -59.1)

        # Cross/marker visuals
        self.cross_len      = rospy.get_param("~cross_half_length", 0.25)
        self.cross_width    = rospy.get_param("~cross_line_width", 0.05)

        # Trajectory controls
        self.traj_enable    = rospy.get_param("~traj_enable", True)
        self.traj_min_dist  = rospy.get_param("~traj_min_dist", 0.03)   # add a point if moved ≥ this (m)
        self.traj_max_pts   = rospy.get_param("~traj_max_points", 5000) # cap to avoid unbounded growth
        self.traj_width     = rospy.get_param("~traj_line_width", 0.03)

        # Republish rate (helps RViz restarts)
        self.repub_hz       = rospy.get_param("~republish_rate_hz", 2.0)

        # ---- State -----------------------------------------------------------
        self.last_true_xyz       = (self.init_x, self.init_y, self.init_z)
        self.last_estimate_xyz   = None
        self.have_model_pose     = False

        # Trajectory buffers (list of (x,y,z))
        self.true_traj = []
        self.est_traj  = []

        # ---- Pub/Sub ---------------------------------------------------------
        # Single latched topic; we publish multiple markers with unique IDs
        self.pub = rospy.Publisher("/acoustic_markers", Marker, queue_size=1, latch=True)

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb, queue_size=1)
        rospy.Subscriber("/acoustic_source_estimate", PoseStamped, self.estimate_cb, queue_size=1)

        # Kick off with initial “true” marker so something is visible early
        self._maybe_append_true_traj(self.last_true_xyz, force_first=True)
        self.publish_true_source_marker(self.last_true_xyz)
        self.publish_true_traj_marker()

        # Timer: republish latest markers periodically
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.repub_hz, 0.01)), self.republish)

        rospy.loginfo("acoustic_source_markers node is up. Tracking model: '%s'", self.model_name)

    # ---- Callbacks -----------------------------------------------------------
    def model_states_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            if not self.have_model_pose:
                rospy.logwarn_throttle(5.0, "[acoustic_source_markers] '%s' not in /gazebo/model_states yet...", self.model_name)
            return

        p = msg.pose[idx].position
        self.last_true_xyz = (p.x, p.y, p.z)
        self.have_model_pose = True

        self._maybe_append_true_traj(self.last_true_xyz)
        self.publish_true_source_marker(self.last_true_xyz)
        self.publish_true_traj_marker()

    def estimate_cb(self, msg: PoseStamped):
        self.last_estimate_xyz = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self._maybe_append_est_traj(self.last_estimate_xyz)
        self.publish_estimate_cross(self.last_estimate_xyz)
        self.publish_est_traj_marker()

    def republish(self, _):
        # Re-publish latest known markers so RViz always shows current state.
        if self.last_true_xyz is not None:
            self.publish_true_source_marker(self.last_true_xyz)
        if self.last_estimate_xyz is not None:
            self.publish_estimate_cross(self.last_estimate_xyz)

        # Re-publish trajectories too
        self.publish_true_traj_marker()
        self.publish_est_traj_marker()

    # ---- Trajectory helpers --------------------------------------------------
    def _maybe_append_true_traj(self, xyz, force_first=False):
        if not self.traj_enable:
            return
        if force_first or not self.true_traj:
            self.true_traj.append(xyz)
            return
        if _dist(xyz, self.true_traj[-1]) >= self.traj_min_dist:
            self.true_traj.append(xyz)
            if len(self.true_traj) > self.traj_max_pts:
                self.true_traj = self.true_traj[-self.traj_max_pts:]

    def _maybe_append_est_traj(self, xyz, force_first=False):
        if not self.traj_enable:
            return
        if force_first or not self.est_traj:
            self.est_traj.append(xyz)
            return
        if _dist(xyz, self.est_traj[-1]) >= self.traj_min_dist:
            self.est_traj.append(xyz)
            if len(self.est_traj) > self.traj_max_pts:
                self.est_traj = self.est_traj[-self.traj_max_pts:]

    # ---- Marker builders -----------------------------------------------------
    def publish_true_source_marker(self, xyz):
        x, y, z = xyz
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "acoustic_source"
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3  # sphere diameter (m)
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.75
        m.lifetime = rospy.Duration(0.0)
        self.pub.publish(m)

    def publish_true_traj_marker(self):
        if not self.traj_enable or len(self.true_traj) < 2:
            return
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "acoustic_source"
        m.id = 11
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = self.traj_width
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8
        m.lifetime = rospy.Duration(0.0)
        m.points = [Point(x, y, z) for (x, y, z) in self.true_traj]
        self.pub.publish(m)

    def publish_estimate_cross(self, xyz):
        x, y, z = xyz
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "acoustic_source"
        m.id = 2
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = self.cross_width  # line width
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.lifetime = rospy.Duration(0.0)

        L = self.cross_len
        # X arm
        m.points.append(Point(x - L, y, z))
        m.points.append(Point(x + L, y, z))
        # Y arm
        m.points.append(Point(x, y - L, z))
        m.points.append(Point(x, y + L, z))
        # Z arm
        m.points.append(Point(x, y, z - L))
        m.points.append(Point(x, y, z + L))

        self.pub.publish(m)

    def publish_est_traj_marker(self):
        if not self.traj_enable or len(self.est_traj) < 2:
            return
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "acoustic_source"
        m.id = 12
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = self.traj_width
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 0.8
        m.lifetime = rospy.Duration(0.0)
        m.points = [Point(x, y, z) for (x, y, z) in self.est_traj]
        self.pub.publish(m)

if __name__ == "__main__":
    try:
        AcousticSourceMarkers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
