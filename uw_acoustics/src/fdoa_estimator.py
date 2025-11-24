#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray, Int64

class FDOAEstimator:
    def __init__(self):
        rospy.init_node("fdoa_estimator", anonymous=False)

        # ---- Parameters ----
        self.f0 = float(rospy.get_param("~f0_hz", 150.0))             # expected tone
        self.fs = float(rospy.get_param("~fs_hz", 100000.0))          # sampling rate of blocks
        self.search_bw = float(rospy.get_param("~search_band_hz", 60.0))  # +/- around f0
        self.min_block = int(rospy.get_param("~min_block", 64))       # minimum usable samples
        self.zero_pad = int(rospy.get_param("~zero_pad", 4096))       # FFT length (power of 2 recommended)
        self.hop_debug = bool(rospy.get_param("~debug_log", False))

        # ---- State ----
        self.last_block_idx_static = None
        self.last_block_idx_rov = None
        self.f_static = np.nan
        self.f_rov = np.nan

        # ---- Publishers ----
        self.pub_f_static = rospy.Publisher("/hydrophones/freq_static_hz", Float32, queue_size=10)
        self.pub_f_rov    = rospy.Publisher("/hydrophones/freq_rov_hz", Float32, queue_size=10)
        self.pub_fdoa     = rospy.Publisher("/hydrophones/fdoa_hz", Float32, queue_size=10)

        # ---- Subscribers ----
        # Expecting Float32MultiArray with a block per message; data_offset carries block index
        rospy.Subscriber("/hydrophones/static", Float32MultiArray, self.cb_static, queue_size=50)
        rospy.Subscriber("/hydrophones/rov",    Float32MultiArray, self.cb_rov,    queue_size=50)

        rospy.loginfo(f"[fdoa_estimator] f0={self.f0} Hz fs={self.fs} Hz search±{self.search_bw} Hz, Nfft={self.zero_pad}")

    # --------------------------
    # Helpers
    # --------------------------
    def estimate_tone_freq(self, x):
        """
        Estimate the dominant tone near f0 using windowed zero-padded FFT + parabolic peak interpolation.
        Returns np.nan on failure.
        """
        x = np.asarray(x, dtype=float)
        N = x.size
        if N < self.min_block:
            return np.nan

        # Remove DC & apply Hann window (reduces leakage)
        x = x - np.mean(x)
        win = np.hanning(N)
        xw = x * win

        # Zero-pad to desired FFT length
        Nfft = int(max(self.zero_pad, 1 << (int(np.ceil(np.log2(N))) )))  # at least power-of-2≥N
        X = np.fft.rfft(xw, n=Nfft)
        freqs = np.fft.rfftfreq(Nfft, d=1.0/self.fs)

        # Limit search to [f0 - bw, f0 + bw], clamp to valid range
        f_lo = max(0.0, self.f0 - self.search_bw)
        f_hi = min(self.fs/2.0, self.f0 + self.search_bw)
        i_lo = int(np.searchsorted(freqs, f_lo))
        i_hi = int(np.searchsorted(freqs, f_hi))
        if i_hi - i_lo < 3:
            return np.nan

        mag = np.abs(X[i_lo:i_hi])
        if mag.size < 3:
            return np.nan

        k = int(np.argmax(mag))
        # Parabolic interpolation around the peak bin (k-1,k,k+1)
        if 0 < k < mag.size - 1:
            y1, y2, y3 = mag[k-1], mag[k], mag[k+1]
            denom = (y1 - 2*y2 + y3)
            if denom != 0.0:
                delta = 0.5 * (y1 - y3) / denom   # fractional bin offset in [-0.5, 0.5]
            else:
                delta = 0.0
        else:
            delta = 0.0

        k_global = i_lo + k
        f_est = (k_global + delta) * (self.fs / Nfft)
        return float(f_est)

    def handle_block(self, x, which, block_idx):
        """
        x: np.ndarray of samples (Float32MultiArray.data)
        which: 'static' or 'rov'
        block_idx: carried in layout.data_offset (int), if available
        """
        fhat = self.estimate_tone_freq(x)
        if np.isfinite(fhat):
            print(f"[fdoa_estimator] {which} block {block_idx} fhat={fhat:.3f} Hz")
            if which == 'static':
                self.f_static = fhat
                self.pub_f_static.publish(Float32(self.f_static))
            else:
                self.f_rov = fhat
                self.pub_f_rov.publish(Float32(self.f_rov))

            if self.hop_debug:
                rospy.loginfo_throttle(1.0, f"[fdoa_estimator] {which} f≈{fhat:.3f} Hz (block {block_idx})")

            # If both are valid, publish FDOA (rov - static)
            if np.isfinite(self.f_static) and np.isfinite(self.f_rov):
                df = float(self.f_rov - self.f_static)
                self.pub_fdoa.publish(Float32(df))
        else:
            # Keep publishing the last good FDOA if desired—or just stay quiet
            if self.hop_debug:
                rospy.logwarn_throttle(2.0, f"[fdoa_estimator] {which} freq estimate failed (block {block_idx})")

    # --------------------------
    # Callbacks
    # --------------------------
    def cb_static(self, msg: Float32MultiArray):
        block_idx = int(msg.layout.data_offset) if msg.layout.data_offset else -1
        # Optionally drop duplicates
        if self.last_block_idx_static is not None and block_idx == self.last_block_idx_static:
            return
        self.last_block_idx_static = block_idx
        self.handle_block(np.array(msg.data, dtype=float), 'static', block_idx)

    def cb_rov(self, msg: Float32MultiArray):
        block_idx = int(msg.layout.data_offset) if msg.layout.data_offset else -1
        if self.last_block_idx_rov is not None and block_idx == self.last_block_idx_rov:
            return
        self.last_block_idx_rov = block_idx
        self.handle_block(np.array(msg.data, dtype=float), 'rov', block_idx)

if __name__ == "__main__":
    try:
        FDOAEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
