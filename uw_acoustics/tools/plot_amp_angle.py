#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

if len(sys.argv) < 2:
    print("Usage: python plot_amp_angle.py /tmp/amp_angle_run.npz")
    sys.exit(1)

path = sys.argv[1]
data = np.load(path)

t          = data["t"]              # time or block index
amp_static = data["amp_static"]
amp_rov    = data["amp_rov"]
angle_deg  = data["angle_deg"]

print("Loaded", path)
print("Num blocks:", t.size)
print("Time span:", float(t[-1] - t[0]) if t.size > 1 else 0.0, "units")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Amplitudes
ax1.plot(t, amp_static, label="Static hydro (mean |amp|)", linewidth=1.8)
ax1.plot(t, amp_rov,    label="ROV hydro (mean |amp|)",   linewidth=1.8, alpha=0.8)
ax1.set_ylabel("Mean |amplitude|")
ax1.grid(True)
ax1.legend()

# Angle
ax2.plot(t, angle_deg, linewidth=0.8)
ax2.set_ylabel("Angle (deg)")
ax2.set_xlabel("Time (s)" if np.any(t > 1.0) else "Block index")
ax2.grid(True)

# fig.suptitle("Hydrophone amplitude vs source→ROV angle")
plt.tight_layout()
plt.show()
