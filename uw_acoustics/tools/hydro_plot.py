#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from datetime import datetime

MAX_BLOCKS = 1000   # just a few blocks for debugging

left_blocks = []
rov_blocks  = []

def cb_left(msg):
    global left_blocks
    arr = np.asarray(msg.data, dtype=np.float32)
    rospy.loginfo("LEFT: got block of size %d, any NaN? %s  min=%.3f max=%.3f",
                  arr.size, np.isnan(arr).any(), float(np.nanmin(arr)), float(np.nanmax(arr)))
    left_blocks.append(arr)
    if len(left_blocks) >= MAX_BLOCKS and len(rov_blocks) >= MAX_BLOCKS:
        rospy.signal_shutdown("Got enough blocks")

def cb_rov(msg):
    global rov_blocks
    arr = np.asarray(msg.data, dtype=np.float32)
    rospy.loginfo("ROV:  got block of size %d, any NaN? %s  min=%.3f max=%.3f",
                  arr.size, np.isnan(arr).any(), float(np.nanmin(arr)), float(np.nanmax(arr)))
    rov_blocks.append(arr)
    if len(left_blocks) >= MAX_BLOCKS and len(rov_blocks) >= MAX_BLOCKS:
        rospy.signal_shutdown("Got enough blocks")

def main():
    rospy.init_node("hydro_debug_recorder")

    topic_left = rospy.get_param("~topic_left", "/hydrophones/static")
    topic_rov  = rospy.get_param("~topic_rov",  "/hydrophones/rov")
    fs         = float(rospy.get_param("~fs", 48000.0))

    rospy.loginfo("Subscribing to %s and %s", topic_left, topic_rov)

    rospy.Subscriber(topic_left, Float32MultiArray, cb_left, queue_size=100)
    rospy.Subscriber(topic_rov,  Float32MultiArray, cb_rov,  queue_size=100)

    try:
        rospy.spin()
    finally:
        if not left_blocks and not rov_blocks:
            rospy.logwarn("No data received, not saving.")
            return

        # concatenate (truncate to same length if needed)
        L = np.concatenate(left_blocks) if left_blocks else np.array([], dtype=np.float32)
        R = np.concatenate(rov_blocks) if rov_blocks  else np.array([], dtype=np.float32)
        n = min(L.size, R.size)
        L = L[:n]
        R = R[:n]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_npz = f"/tmp/hydro_debug_{ts}.npz"
        np.savez(out_npz, left=L, rov=R, fs=np.array([fs], dtype=np.float64))

        rospy.loginfo("Saved %s samples to %s (any NaN? L=%s R=%s)",
                      n, out_npz,
                      np.isnan(L).any(), np.isnan(R).any())

if __name__ == "__main__":
    main()
