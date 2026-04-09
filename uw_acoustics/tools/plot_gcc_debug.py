#!/usr/bin/env python3

# =====================================
# Author : Adnan Abdullah
# Email: adnanabdullah@ufl.edu
# =====================================


import argparse, numpy as np, matplotlib.pyplot as plt

def next_pow2(n): return 1 << (int(n) - 1).bit_length()

# ---------- GCC-PHAT (broadband) ----------
def gcc_phat_block(x, y, fs, bp=(200,4000), interp=8, max_tau=None, avg_Rxy=None, alpha=0.0):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    x -= x.mean(); y -= y.mean()
    Nblk = x.size
    Nfft = next_pow2(2 * Nblk)

    X = np.fft.rfft(x, n=Nfft)
    Y = np.fft.rfft(y, n=Nfft)
    Rxy = X * np.conj(Y)
    mag = np.abs(Rxy)
    Rxy = np.where(mag > 1e-15, Rxy / mag, 0.0)

    freqs = np.fft.rfftfreq(Nfft, d=1.0/fs)
    if bp is not None:
        lo, hi = bp
        mask = (freqs >= lo) & (freqs <= hi)
        Rxy *= mask.astype(np.float64)

    if avg_Rxy is not None and alpha > 0.0:
        Rxy = alpha * avg_Rxy + (1.0 - alpha) * Rxy
        avg_Rxy[:] = Rxy

    cc = np.fft.irfft(Rxy, n=Nfft * interp)
    cc = np.fft.fftshift(cc)  # center zero-lag
    L = cc.size
    k = np.arange(-L//2, L//2 + (L % 2))
    lags_sec = k / float(interp * fs)

    if max_tau is not None:
        keep = np.abs(lags_sec) <= max_tau
        cc = cc[keep]; lags_sec = lags_sec[keep]

    i_peak = int(np.argmax(cc))
    tau = float(lags_sec[i_peak])
    return tau, cc, lags_sec, freqs, np.abs(X), np.abs(Y)

# ---------- Tone phase (narrowband) ----------
def tone_phase_delay(x, y, fs, k_halfwidth=3):
    """
    Estimate TDOA from cross-spectrum phase around the dominant tone.
    Returns principal-value tau (sec), tone period T (sec), tone freq f0 (Hz).
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    x -= x.mean(); y -= y.mean()
    Nblk = x.size
    Nfft = next_pow2(2 * Nblk)
    X = np.fft.rfft(x, n=Nfft)
    Y = np.fft.rfft(y, n=Nfft)

    # find dominant bin (ignore DC)
    mag_sum = np.abs(X) + np.abs(Y)
    k0 = int(np.argmax(mag_sum[1:])) + 1

    k1 = max(1, k0 - k_halfwidth)
    k2 = min(len(X) - 1, k0 + k_halfwidth)
    k = np.arange(k1, k2 + 1)
    freqs = k * fs / Nfft

    R = X[k] * np.conj(Y[k])
    w = np.abs(R) + 1e-15
    Rw = np.sum(R * w) / np.sum(w)            # weighted cross-spectrum
    f0 = float(np.sum(freqs * w) / np.sum(w)) # weighted center freq

    phi = np.angle(Rw)                         # cross phase (Y vs X)
    tau_principal = -phi / (2.0 * np.pi * f0) # principal value in (−T/2, T/2]
    T = 1.0 / f0
    return tau_principal, T, f0, k0

def unwrap_tone_tau(tau_principal, T, max_tau, tau_hint=None):
    """
    Unwrap tone delay using multiples of T to land within ±max_tau.
    If tau_hint is provided (sec), choose the candidate nearest to it.
    Otherwise choose the candidate with smallest |tau|.
    """
    # enumerate candidates within ±max_tau
    mmin = int(np.floor((-max_tau - tau_principal) / T))
    mmax = int(np.ceil( ( max_tau - tau_principal) / T))
    cands = [tau_principal + m*T for m in range(mmin, mmax+1)]
    if tau_hint is not None:
        best = min(cands, key=lambda t: abs(t - tau_hint))
    else:
        best = min(cands, key=lambda t: abs(t))
    return best, cands

# ---------- CLI / plotting ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="hydro_dump.npz")
    ap.add_argument("--block", type=int, default=None,
                    help="block index to plot (0-based). Default: average all (for PHAT).")
    ap.add_argument("--avg", action="store_true",
                    help="exponentially average PHAT cross-spectrum across all blocks")
    ap.add_argument("--alpha", type=float, default=0.9,
                    help="EMA alpha for PHAT averaging (if --avg)")
    ap.add_argument("--bp", type=str, default="200,4000",
                    help="PHAT bandpass Hz 'lo,hi' or 'none'")
    ap.add_argument("--interp", type=int, default=8)
    ap.add_argument("--invert_sign", action="store_true",
                    help="flip tau sign (if left/right order opposite)")
    ap.add_argument("--tau_hint_us", type=float, default=None,
                    help="hint (µs) to unwrap tone TDOA to the correct branch")
    args = ap.parse_args()

    D = np.load(args.npz)
    L = D["left"]; R = D["right"]; idx = D["idx"]
    fs = float(D["fs"]); b = float(D["baseline_m"]); c = float(D["c_sound"])
    Nblocks, Nblk = L.shape
    tau_max = b / c

    print(f"Loaded {args.npz}: blocks={Nblocks}, N={Nblk}, fs={fs}, b={b}, c={c}, tau_max={tau_max*1e6:.1f} µs")

    bp = None if args.bp.lower()=="none" else tuple(map(float, args.bp.split(",")))

    # choose blocks (for display)
    chosen = range(Nblocks) if args.block is None else [args.block]

    # PHAT (optional average across chosen)
    avg_Rxy = None
    if args.avg:
        Nfft = next_pow2(2 * Nblk)
        avg_Rxy = np.zeros(Nfft//2 + 1, dtype=np.complex128)

    phat_taus = []
    last_phat = None
    for k in chosen:
        tau_p, cc, lags, freqs, Xmag, Ymag = gcc_phat_block(
            L[k], R[k], fs, bp=bp, interp=args.interp,
            max_tau=tau_max*1.2, avg_Rxy=avg_Rxy, alpha=args.alpha if args.avg else 0.0)
        if args.invert_sign: tau_p = -tau_p
        phat_taus.append(tau_p)
        last_phat = dict(cc=cc, lags=lags, freqs=freqs, Xmag=Xmag, Ymag=Ymag, k=k)

    # Tone phase (single block for display)
    k_disp = last_phat["k"]
    tau_pr, T, f0, _ = tone_phase_delay(L[k_disp], R[k_disp], fs)
    if args.invert_sign: tau_pr = -tau_pr
    tau_hint = args.tau_hint_us/1e6 if args.tau_hint_us is not None else None
    tau_tone, tone_candidates = unwrap_tone_tau(tau_pr, T, tau_max*1.2, tau_hint=tau_hint)

    # Summaries
    tau_phat = float(np.mean(phat_taus))
    smp_phat = tau_phat * fs
    doa_phat = float(np.degrees(np.arcsin(np.clip(c * tau_phat / b, -1.0, 1.0))))

    smp_tone = tau_tone * fs
    doa_tone = float(np.degrees(np.arcsin(np.clip(c * tau_tone / b, -1.0, 1.0))))

    print(f"PHAT:   tau = {tau_phat*1e6:.2f} µs  ({smp_phat:.2f} samples),  DoA ≈ {doa_phat:.2f}°")
    print(f"TONE:   tau_principal = {tau_pr*1e6:.2f} µs  (T = {1e6*T:.2f} µs, f0 ≈ {f0:.2f} Hz)")
    print(f"        unwrapped tau = {tau_tone*1e6:.2f} µs  ({smp_tone:.2f} samples),  DoA ≈ {doa_tone:.2f}°")
    if args.tau_hint_us is None:
        print("        (pass --tau_hint_us to pick the physically correct branch for a tone)")

    # ---- Plot (using the last PHAT block and the tone result) ----
    t = np.arange(Nblk) / fs
    x = L[k_disp]; y = R[k_disp]

    fig = plt.figure(figsize=(12,10))

    ax1 = plt.subplot(2,2,1)
    ax1.plot(t, x, label="left")
    ax1.plot(t, y, label="right", alpha=0.85)
    ax1.set_title(f"Time domain (block {k_disp}, N={Nblk})")
    ax1.set_xlabel("Time [s]"); ax1.legend()

    ax2 = plt.subplot(2,2,2)
    ax2.semilogy(last_phat["freqs"], last_phat["Xmag"] + 1e-12, label="|X|")
    ax2.semilogy(last_phat["freqs"], last_phat["Ymag"] + 1e-12, label="|Y|", alpha=0.85)
    ax2.set_title("Magnitude spectra"); ax2.set_xlabel("Frequency [Hz]"); ax2.legend()

    ax3 = plt.subplot(2,1,2)
    ax3.plot(last_phat["lags"]*1e6, last_phat["cc"], label="GCC-PHAT")
    ax3.axvline(tau_phat*1e6, color='k', linestyle='--',
                label=f"PHAT: τ={tau_phat*1e6:.2f} µs ({smp_phat:.2f} smp) | DoA≈{doa_phat:.2f}°")
    # tone markers (all candidates), and the selected unwrapped one
    for cand in tone_candidates:
        ax3.axvline(cand*1e6, color='C1', alpha=0.25)
    ax3.axvline(tau_tone*1e6, color='C1', linestyle='--',
                label=f"TONE: τ={tau_tone*1e6:.2f} µs ({smp_tone:.2f} smp) | DoA≈{doa_tone:.2f}°")
    ax3.set_title("PHAT cross-correlation and tone candidates")
    ax3.set_xlabel("Lag [µs]"); ax3.grid(True); ax3.legend()

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
