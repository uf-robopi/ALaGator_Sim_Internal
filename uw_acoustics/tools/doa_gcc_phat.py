#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import rospy
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray, String

def next_pow2(n): return 1 << (int(n) - 1).bit_length()

def build_bp_mask(Nfft, fs, lo, hi):
    freqs = np.fft.rfftfreq(Nfft, d=1.0/fs)
    return ((freqs >= lo) & (freqs <= hi)).astype(np.float64)

def time_domain_integer_lag(x, y, max_lag_samp):
    """Integer-lag via time-domain cross-correlation (no PHAT)."""
    x = x - x.mean(); y = y - y.mean()
    cc_full = np.correlate(x, y, mode='full')      # len = 2N-1
    lags = np.arange(-x.size+1, x.size)
    keep = (np.abs(lags) <= max_lag_samp)
    cc = cc_full[keep]; k = lags[keep]
    return int(k[int(np.argmax(cc))])


def gcc_integer_lag(x, y, fs, max_lag_samp, use_phat=True,
                    Nfft=None, bp_mask=None, alpha_spec=0.0, Rxy_avg=None):
    """
    Integer-lag GCC:
      - mean-remove, FFT, optional PHAT weighting + band-limit
      - optional EMA on cross-spectrum
      - IFFT -> correlation; pick integer lag in [-max_lag_samp, +max_lag_samp]
    Returns: (lag_samples:int, Rxy_avg:new_or_updated)
    """
    x = x - x.mean()
    y = y - y.mean()
    Nblk = x.size
    if Nfft is None:
        Nfft = next_pow2(2 * Nblk)

    X = np.fft.rfft(x, n=Nfft)
    Y = np.fft.rfft(y, n=Nfft)

    if use_phat:
        Rxy = X * np.conj(Y)
        mag = np.abs(Rxy)
        Rxy = np.where(mag > 1e-15, Rxy / mag, 0.0)
    else:
        Rxy = X * np.conj(Y)

    if bp_mask is not None:
        Rxy *= bp_mask

    if alpha_spec > 0.0:
        if Rxy_avg is None:
            Rxy_avg = Rxy.astype(np.complex128)
        else:
            Rxy_avg[:] = alpha_spec * Rxy_avg + (1.0 - alpha_spec) * Rxy
        Ruse = Rxy_avg
    else:
        Ruse = Rxy

    cc = np.fft.irfft(Ruse, n=Nfft)       # correlation (circular)
    cc = np.fft.fftshift(cc)              # center zero-lag
    L = cc.size
    k = np.arange(-L//2, L//2 + (L % 2))  # integer lags (samples), centered

    keep = (np.abs(k) <= max_lag_samp)
    cc_win = cc[keep]
    k_win  = k[keep]

    lag_samples = int(k_win[int(np.argmax(cc_win))])
    return lag_samples, Rxy_avg


def windowed_period_correlation(sig1, sig2, fs, f, sound_speed=1482.0):
    """
    Windowed correlation over a single period to estimate sample lag, phase, and TDOA (time difference of arrival).

    Parameters
    ----------
    sig1, sig2 : 1D np.ndarray
        Input signals. Expected length ≈ 3T where T = 1/f.
    fs : float
        Sampling frequency [Hz].
    f : float
        Signal frequency [Hz].
    sound_speed : float
        Propagation speed (for path difference) [m/s]. Default 1482 (water).
    plot : bool
        If True, plots original signals (with window highlighted) and correlation vs. lag.

    Returns
    -------
    result : dict
        {
            'corr': np.ndarray,           # normalized correlation values over scanned lags
            'lags': np.ndarray,           # lag indices (samples)
            'lag_samples': int,           # peak lag in samples (sig2 shifted by this aligns with sig1)
            'time_shift': float,          # peak lag as time (seconds); <0 means sig2 leads
            'phase_deg': float,           # estimated phase shift (degrees), wrapped to (-180, 180]
            'TDOA': float,                # time difference of arrival (micro seconds) = time_shift
            'path_diff': float,           # Δd = sound_speed * TDOA (meters)
            'P': int,                     # samples per period
            'window_idx': (int, int)      # [start, end) indices for the fixed window on sig1
        }
    """
    sig1 = np.asarray(sig1).astype(float)
    sig2 = np.asarray(sig2).astype(float)
    assert sig1.ndim == 1 and sig2.ndim == 1 and len(sig1) == len(sig2), "Signals must be same-length 1D arrays."

    T = 1.0 / f
    P = int(round(fs * T))                 # samples per period
    N = len(sig1)
    assert N >= 3 * P, "Signals should be at least 3 periods long (≈ 3T samples)."

    # Use the middle period of sig1 as fixed window: indices [P, 2P)
    x_start, x_end = P, 2 * P
    x_win = sig1[x_start:x_end]

    # Lags within ±P/2 to ensure full overlap of one-period windows within 3T
    halfP = P // 2
    lags = np.arange(-halfP, halfP + 1)

    # Precompute norm
    x_norm = np.sqrt(np.sum(x_win**2)) + 1e-12

    corr = np.empty_like(lags, dtype=float)
    for i, k in enumerate(lags):
        y_start = x_start + k
        y_end   = x_end   + k
        y_win   = sig2[y_start:y_end]
        y_norm = np.sqrt(np.sum(y_win**2)) + 1e-12
        corr[i] = np.sum(x_win * y_win) / (x_norm * y_norm)

    # Peak lag -> time shift
    peak_idx = int(np.argmax(corr))
    lag_samples = int(lags[peak_idx])
    time_shift = lag_samples / fs   # negative => sig2 leads sig1

    # Phase estimate with correct sign convention:
    # sig2(t) = sin(ωt + φ). If sig2 leads (φ>0), peak lag is negative.
    # φ ≈ -ω * time_shift
    phase_rad = (-2*np.pi*f*time_shift + np.pi) % (2*np.pi) - np.pi
    phase_deg = np.degrees(phase_rad)

    # TDOA and path difference
    TDOA = time_shift
    path_diff = sound_speed * TDOA

    result = {
        'corr': corr,
        'lags': lags,
        'lag_samples': lag_samples,
        'time_shift': time_shift,
        'phase_deg': phase_deg,
        'TDOA': TDOA * 1e6,  # convert to microseconds
        'path_diff': path_diff,
        'P': P,
        'window_idx': (x_start, x_end),
    }

    return result


class DoANode(object):
    def __init__(self):
        # Topics / geometry
        self.topic_left  = rospy.get_param("~topic_left",  "/hydrophones/left")
        self.topic_right = rospy.get_param("~topic_right", "/hydrophones/right")
        self.fs          = float(rospy.get_param("~fs", 100000.0))
        self.baseline_m  = float(rospy.get_param("~baseline_m", 0.5))
        self.c_sound     = float(rospy.get_param("~speed_of_sound", 1482.0))

        # Integer-lag GCC controls
        self.use_phat_int = bool(rospy.get_param("~use_phat_int", True))   # PHAT weighting
        self.alpha_spec   = float(rospy.get_param("~alpha_spec", 0.90))    # EMA on cross-spectrum (0..0.99)
        self.bp_lo        = float(rospy.get_param("~bp_lo", 200.0))
        self.bp_hi        = float(rospy.get_param("~bp_hi", 8000.0))

        # Physical bounds & sign
        self.max_tau_factor = float(rospy.get_param("~max_tau_factor", 1.0))  # 1.0 = physical max
        self.invert_sign    = bool(rospy.get_param("~invert_sign", False))

        # Bearing smoothing (optional)
        self.use_ema   = bool(rospy.get_param("~use_ema", True))
        self.ema_alpha = float(rospy.get_param("~ema_alpha", 0.8))

        # Derived
        self.tau_max_phys = self.baseline_m / self.c_sound          # seconds
        self.max_tau      = self.tau_max_phys * self.max_tau_factor
        self.max_lag_samp = int(np.floor(self.max_tau * self.fs))   # integer samples

        # Pairing buffer: idx -> {"L": np.array, "R": np.array}
        self.blocks = {}

        # Spectral state
        self.Nfft    = None
        self.bp_mask = None
        self.Rxy_avg = None

        # Publishers
        self.pub_angle   = rospy.Publisher("~bearing_deg", Float32, queue_size=20)
        self.pub_tau_ms  = rospy.Publisher("~tdoa_ms", Float32, queue_size=20)
        self.pub_tau_smp = rospy.Publisher("~tdoa_samples", Float32, queue_size=20)
        self.pub_method  = rospy.Publisher("~method", String, queue_size=1, latch=True)

        # Subscribers (read idx from layout.data_offset; big queues to avoid drops)
        rospy.Subscriber(self.topic_left,  Float32MultiArray, self.cb_audio, queue_size=200, callback_args="L")
        rospy.Subscriber(self.topic_right, Float32MultiArray, self.cb_audio, queue_size=200, callback_args="R")

        self._angle_ema = None
        self.pub_method.publish("integral")  # advertised mode label
        rospy.loginfo("[doa_integral] fs=%.1f Hz, baseline=%.3f m, c=%.1f m/s, max lag ≈ %d samples",
                      self.fs, self.baseline_m, self.c_sound, self.max_lag_samp)

    # -------- callbacks --------
    def cb_audio(self, msg, side):
        idx = int(msg.layout.data_offset)              # block index embedded by plugin
        arr = np.asarray(msg.data, dtype=np.float64)
        slot = self.blocks.get(idx, {"L": None, "R": None})
        slot["L" if side == "L" else "R"] = arr
        self.blocks[idx] = slot
        if slot["L"] is not None and slot["R"] is not None:
            self.process(idx)

    # -------- processing --------
    def _init_spectral(self, Nblk):
        self.Nfft = next_pow2(2 * Nblk)
        self.bp_mask = build_bp_mask(self.Nfft, self.fs, self.bp_lo, self.bp_hi)
        self.Rxy_avg = None

    def process(self, idx):
        x = self.blocks[idx]["L"]; y = self.blocks[idx]["R"]
        del self.blocks[idx]
        if x is None or y is None or x.size == 0 or y.size == 0 or x.size != y.size:
            return
        if self.Nfft is None:
            self._init_spectral(x.size)

        # Integer-lag GCC
        if self.use_phat_int:
            lag_samp, self.Rxy_avg = gcc_integer_lag(
                x, y, self.fs, self.max_lag_samp,
                use_phat=self.use_phat_int,
                Nfft=self.Nfft, bp_mask=self.bp_mask,
                alpha_spec=self.alpha_spec, Rxy_avg=self.Rxy_avg
            )
        else:
            # print("Using windowed period correlation")
            # lag_samp = time_domain_integer_lag(x, y, self.max_lag_samp)
            result = windowed_period_correlation(x, y, self.fs, f=1500, sound_speed=self.c_sound)
            lag_samp = result['lag_samples']

        tau = lag_samp / float(self.fs)

        # clamp + sign
        tau = float(np.clip(tau, -self.tau_max_phys, self.tau_max_phys))
        if self.invert_sign: tau = -tau

        # map to bearing (far-field)
        s = (self.c_sound * tau) / self.baseline_m
        s = float(np.clip(s, -1.0, 1.0))
        angle = float(np.degrees(np.arcsin(s)))

        # EMA (optional)
        if self.use_ema:
            self._angle_ema = angle if self._angle_ema is None else \
                              self.ema_alpha*self._angle_ema + (1.0-self.ema_alpha)*angle
            angle = self._angle_ema

        rospy.loginfo_throttle(1.0, "[doa] lag=%d samp  tau=%.2fus  bearing=%.2f deg",
                       int(lag_samp), 1e6*(lag_samp/self.fs), angle)

        # Publish
        self.pub_angle.publish(Float32(data=angle))
        self.pub_tau_ms.publish(Float32(data=1e3 * tau))
        self.pub_tau_smp.publish(Float32(data=float(lag_samp)))

if __name__ == "__main__":
    rospy.init_node("doa_gcc_phat")
    DoANode()
    rospy.loginfo("[doa_integral] ready")
    rospy.spin()
