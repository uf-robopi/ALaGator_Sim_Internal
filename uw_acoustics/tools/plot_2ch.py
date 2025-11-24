#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
# ==== USER SETTINGS ====
filename = "/home/adnana/catkin_ws/hydro_dump_n_0_01_noloss.npz"   # Path to saved npz file
start_block = 1                # First block index to include
end_block   = 1                # Last block index (inclusive)
# ========================

# Load file
D = np.load(filename)
L = D["left"]         # shape: (n_blocks, block_size)
R = D["right"]        # shape: (n_blocks, block_size)
idx = D["idx"]        # shape: (n_blocks,)
fs = float(D["fs"])   # sampling rate [Hz]

# Clamp to valid range
end_block = min(end_block, len(idx) - 1)

# Concatenate selected blocks
L_concat = np.concatenate(L[start_block:end_block+1])
R_concat = np.concatenate(R[start_block:end_block+1])

# Create continuous time vector
n_samples = len(L_concat)
t = 1e6 * np.arange(n_samples) / fs

# Plot
plt.figure(figsize=(12, 4))
plt.plot(t, L_concat, label=" H1 (Static)")
plt.plot(t, R_concat, label=" H2 (Mobile)", alpha=0.7)
# plt.title(f"Blocks {idx[start_block]} to {idx[end_block]}")
plt.title("Noise Std = 0.01, Loss Model = None")
plt.xlabel("Time [us]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
