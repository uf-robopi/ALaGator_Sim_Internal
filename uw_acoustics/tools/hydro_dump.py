#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================

import rospy, numpy as np, os, time
from std_msgs.msg import Float32MultiArray

class Dumper:
    def __init__(self):
        self.topic_L = rospy.get_param("~topic_left",  "/hydrophones/static")
        self.topic_R = rospy.get_param("~topic_right", "/hydrophones/rov")
        self.Nblocks = int(rospy.get_param("~n_blocks", 32))
        self.outpath = rospy.get_param("~out", "hydro_dump_n_0_05_inverse.npz")
        self.fs      = float(rospy.get_param("~fs", 48000.0))
        self.baseline= float(rospy.get_param("~baseline_m", 2.0))
        self.csound  = float(rospy.get_param("~speed_of_sound", 1482.0))

        # idx -> {"L": arr, "R": arr}
        self.buf = {}
        self.collected = []

        rospy.Subscriber(self.topic_L, Float32MultiArray, self.cb_audio, queue_size=200, callback_args="L")
        rospy.Subscriber(self.topic_R, Float32MultiArray, self.cb_audio, queue_size=200, callback_args="R")
        rospy.loginfo("[hydro_dump] recording %d paired blocks -> %s", self.Nblocks, os.path.abspath(self.outpath))

    def cb_audio(self, msg, side):
        # read block_idx carried in layout.data_offset
        idx = int(msg.layout.data_offset)
        arr = np.asarray(msg.data, dtype=np.float64)
        slot = self.buf.get(idx, {"L": None, "R": None})
        slot["L" if side=="L" else "R"] = arr
        self.buf[idx] = slot

        if slot["L"] is not None and slot["R"] is not None:
            self.collected.append((idx, slot["L"], slot["R"]))
            del self.buf[idx]
            rospy.loginfo_throttle(1.0, "[hydro_dump] paired %d/%d", len(self.collected), self.Nblocks)
            if len(self.collected) >= self.Nblocks:
                self.flush_and_exit()

    def flush_and_exit(self):
        self.collected.sort(key=lambda t: t[0])
        idxs  = np.array([t[0] for t in self.collected], dtype=np.int64)
        lefts = np.stack([t[1] for t in self.collected], axis=0)
        rights= np.stack([t[2] for t in self.collected], axis=0)

        # basic continuity check
        gaps = np.where(np.diff(idxs) != 1)[0]
        if gaps.size:
            rospy.logwarn("[hydro_dump] WARNING: non-contiguous idx at positions %s (idx=%s)", gaps.tolist(), idxs[gaps].tolist())

        np.savez(self.outpath, idx=idxs, left=lefts, right=rights,
                 fs=self.fs, baseline_m=self.baseline, c_sound=self.csound, t_dump=time.time())
        rospy.loginfo("[hydro_dump] wrote %s  (blocks=%d, block_size=%d)", os.path.abspath(self.outpath), lefts.shape[0], lefts.shape[1])
        rospy.signal_shutdown("done")

if __name__ == "__main__":
    rospy.init_node("hydro_dump")
    Dumper()
    rospy.spin()
