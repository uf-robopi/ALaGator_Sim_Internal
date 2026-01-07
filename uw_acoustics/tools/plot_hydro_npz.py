#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({'font.size': 14})

if len(sys.argv) < 2:
    print("Usage: python plot_simple_hydro.py hydro_simple_run.npz")
    sys.exit(1)

data = np.load(sys.argv[1])
L = data["left"]
R = data["rov"]
fs = float(data["fs"][0])

print("Loaded", sys.argv[1], "samples:", L.size, "duration ~", L.size/fs, "s")

t = np.arange(L.size) / fs
plt.figure(figsize=(14,6))
plt.plot(t, L, label="Static H", linewidth=0.7)
plt.plot(t, R, label="Mobile H", linewidth=0.7, alpha=0.8)
plt.grid(True)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
