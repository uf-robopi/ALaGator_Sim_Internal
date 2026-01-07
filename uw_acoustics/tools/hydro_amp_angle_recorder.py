#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from datetime import datetime

class AmpAngleRecorder:
    def __init__(self):
        # Topics
        self.topic_static = rospy.get_param("~topic_static", "/hydrophones/static")
        self.topic_rov    = rospy.get_param("~topic_rov",    "/hydrophones/rov")
        self.fs           = float(rospy.get_param("~fs", 100000.0))

        # Gazebo model names
        self.source_model = rospy.get_param("~source_model", "mobile_acoustic_source")
        self.rov_model    = rospy.get_param("~rov_model",    "nemesys")  # set this appropriately

        # Output file
        out_npz = rospy.get_param("~out_npz", "")
        if not out_npz:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_npz = "/tmp/hydro_amp_angle_{}.npz".format(ts)
        self.out_npz = out_npz

        # Storage
        self.amp_static_blocks = []   # list of floats, one per static block
        self.amp_rov_blocks    = []   # list of floats, one per rov block
        self.angle_blocks      = []   # list of floats (deg), aligned with amp_rov_blocks

        # Geometry cache
        self.source_pos = None  # (x,y,z)
        self.rov_pos    = None
        self.source_yaw = None  # radians

        # Block timing (assume constant block length)
        self.block_dt = None  # seconds per block (len(block)/fs)
        self.block_count_rov = 0

        # Subscribers
        rospy.Subscriber(self.topic_static, Float32MultiArray,
                         self.cb_static, queue_size=200)
        rospy.Subscriber(self.topic_rov, Float32MultiArray,
                         self.cb_rov, queue_size=200)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self.cb_model_states, queue_size=10)

        rospy.loginfo("[hydro_amp_angle_recorder] fs=%.1f, topics: static=%s, rov=%s, out=%s",
                      self.fs, self.topic_static, self.topic_rov, self.out_npz)
        rospy.loginfo("[hydro_amp_angle_recorder] source_model=%s, rov_model=%s",
                      self.source_model, self.rov_model)

    # -------- geometry from Gazebo --------
    def cb_model_states(self, msg):
        # Cache source + rov pose + source yaw (world->body)
        try:
            i_s = msg.name.index(self.source_model)
            i_r = msg.name.index(self.rov_model)
        except ValueError:
            # one or both not spawned yet
            return

        ps = msg.pose[i_s]
        pr = msg.pose[i_r]

        self.source_pos = (ps.position.x, ps.position.y, ps.position.z)
        self.rov_pos    = (pr.position.x, pr.position.y, pr.position.z)

        q = ps.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        self.source_yaw = yaw

    # -------- hydro blocks --------
    def cb_static(self, msg):
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.size == 0:
            return
        amp = float(np.mean(np.abs(arr)))
        self.amp_static_blocks.append(amp)

    def cb_rov(self, msg):
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.size == 0:
            return

        # infer block dt once
        if self.block_dt is None:
            self.block_dt = arr.size / float(self.fs)
            rospy.loginfo("[hydro_amp_angle_recorder] inferred block_size=%d, dt=%.6f s",
                          arr.size, self.block_dt)

        amp = float(np.mean(np.abs(arr)))
        self.amp_rov_blocks.append(amp)

        # angle from source forward to ROV direction
        angle_deg = np.nan
        if self.source_pos is not None and self.rov_pos is not None and self.source_yaw is not None:
            sx, sy, _ = self.source_pos
            rx, ry, _ = self.rov_pos
            dx = rx - sx
            dy = ry - sy
            # bearing of ROV from source in world frame
            theta_world = np.arctan2(dy, dx)
            # relative to source forward (yaw)
            dtheta = theta_world - self.source_yaw
            # wrap to [-pi, pi]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            angle_deg = float(np.degrees(dtheta))

        self.angle_blocks.append(angle_deg)
        self.block_count_rov += 1

    # -------- finalize & save --------
    def finalize(self):
        n_rov = len(self.amp_rov_blocks)
        n_static = len(self.amp_static_blocks)
        n_angle = len(self.angle_blocks)

        if n_rov == 0:
            rospy.logwarn("[hydro_amp_angle_recorder] No ROV blocks recorded; nothing to save.")
            return

        # Align static amplitudes to rov blocks by index: static[i] -> rov[i]
        amp_static = np.full(n_rov, np.nan, dtype=np.float32)
        for i in range(min(n_rov, n_static)):
            amp_static[i] = self.amp_static_blocks[i]

        amp_rov = np.array(self.amp_rov_blocks, dtype=np.float32)
        angle   = np.array(self.angle_blocks,    dtype=np.float32)

        # Time / block index
        block_idx = np.arange(n_rov, dtype=np.int64)
        if self.block_dt is not None:
            t = block_idx * self.block_dt
        else:
            t = block_idx.astype(np.float64)  # unknown dt; treat as “block index”

        rospy.loginfo("[hydro_amp_angle_recorder] saving %d ROV blocks (static=%d, angles=%d) to %s",
                      n_rov, n_static, n_angle, self.out_npz)

        np.savez(self.out_npz,
                 t=t,
                 block_idx=block_idx,
                 amp_static=amp_static,
                 amp_rov=amp_rov,
                 angle_deg=angle,
                 fs=np.array([self.fs], dtype=np.float64),
                 block_dt=np.array([self.block_dt if self.block_dt is not None else -1.0],
                                   dtype=np.float64))
        rospy.loginfo("[hydro_amp_angle_recorder] done.")

def main():
    rospy.init_node("hydro_amp_angle_recorder", anonymous=True)
    rec = AmpAngleRecorder()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rec.finalize()

if __name__ == "__main__":
    main()
