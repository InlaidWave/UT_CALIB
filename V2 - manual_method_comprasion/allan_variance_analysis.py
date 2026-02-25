#!/usr/bin/env python3
"""
allan_vs_tau_rad2_full.py

Compute overlapping Allan variance (rad^2 / s^2) from Arduino logs,
following the Tedaldi-style overlapping definition, and display a linear
plot with one tick per second on the x-axis up to TAU_MAX_REQ.

Usage:
    pip install numpy matplotlib
    python allan_vs_tau_rad2_full.py [path/to/log.txt]

Notes:
 - TAU_MAX_REQ defines the displayed x-axis range (default 300 s).
 - The script computes Allan variance only where raw timestamps give enough
   data; remaining taus are left NaN (gaps in plot). The x-axis still runs
   to TAU_MAX_REQ with one tick per second as you requested.
"""
import sys, re, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = "SERVO_DATA"
TAU_MAX_REQ = 1800  # display 1..300 seconds on x-axis

# ---------------- file chooser ----------------
def choose_file(folder):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        print(f"Folder '{folder}' not found.")
        return None
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower()=='.txt'])
    if not files:
        print(f"No .txt files in '{folder}'.")
        return None
    print(f"Found {len(files)} files:")
    for i,f in enumerate(files):
        print(f" {i}: {f.name} ({f.stat().st_size} bytes)")
    while True:
        s = input("Select file number (q to quit): ").strip()
        if s.lower() == 'q':
            return None
        try:
            idx = int(s)
            if 0 <= idx < len(files):
                return files[idx]
        except Exception:
            pass
        print("Enter a valid index or 'q'.")

# ---------------- parsing ----------------
def parse_log(fn):
    re_t = re.compile(r"[&]?T\s*(\d+)")
    re_g = re.compile(r"[&]?GX\s*([-\d\.eE]+).*?[&]?GY\s*([-\d\.eE]+).*?[&]?GZ\s*([-\d\.eE]+)")
    ts = []; gx=[]; gy=[]; gz=[]
    with open(fn, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.strip(): continue
            mt = re_t.search(ln)
            mg = re_g.search(ln)
            if mt is None or mg is None: continue
            try:
                ts_val = int(mt.group(1))
                gx_val = float(mg.group(1))
                gy_val = float(mg.group(2))
                gz_val = float(mg.group(3))
            except Exception:
                continue
            ts.append(ts_val/1e6)   # µs -> s
            gx.append(gx_val); gy.append(gy_val); gz.append(gz_val)
    return np.array(ts), np.array(gx), np.array(gy), np.array(gz)

# ---------------- uniform resampling ----------------
def resample_uniform(t, x, target_fs=None):
    order = np.argsort(t)
    t = t[order]; x = x[order]
    dt = np.diff(t)
    if dt.size == 0:
        raise ValueError("Not enough samples to resample.")
    median_dt = np.median(dt)
    fs = (1.0/median_dt) if target_fs is None else float(target_fs)
    t0, t1 = t[0], t[-1]
    N = int(np.floor((t1 - t0) * fs)) + 1
    if N < 3:
        raise ValueError("Not enough duration/samples for resampling.")
    t_uniform = t0 + np.arange(N) / fs
    x_uniform = np.interp(t_uniform, t, x)
    return t_uniform, x_uniform, fs, np.mean(dt), np.std(dt), median_dt

# ---------------- overlapping Allan variance (Tedaldi indexing) ----------------
def allan_variance_overlapping_from_uniform(x_uniform, fs, tau_s):
    """
    Overlapping Allan variance for a given tau (seconds).
    x_uniform: evenly sampled series (in rad/s)
    Returns (sigma2, K) where K is number of overlapping differences: N - 2*m + 1
    """
    if tau_s <= 0:
        return None, 0
    m = int(round(tau_s * fs))
    if m < 1:
        m = 1
    N = int(x_uniform.size)
    if 2 * m >= N:
        return None, 0
    kernel = np.ones(m, dtype=float) / float(m)
    mu = np.convolve(x_uniform, kernel, mode='valid')    # length M = N - m + 1
    K = mu.size - m
    if K <= 0:
        return None, 0
    diffs = mu[m:] - mu[:-m]   # length K
    sigma2 = 0.5 * np.mean(diffs * diffs)
    return float(sigma2), int(K)

# ---------------- choose T_init heuristic ----------------
def choose_Tinit_from_sigma(taus, sigma_vals, K_used, tol_rel=1.10, min_K=8):
    mask_valid = (~np.isnan(sigma_vals)) & (sigma_vals > 0)
    if not mask_valid.any():
        return None
    taus_v = taus[mask_valid]
    sigma_v = sigma_vals[mask_valid]
    K_v = K_used[mask_valid]
    sigma_min = np.min(sigma_v)
    threshold = tol_rel * sigma_min
    for tau, s, K in zip(taus_v, sigma_v, K_v):
        if (s <= threshold) and (K >= min_K):
            return int(tau)
    return None

# ---------------- main ----------------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        INPUT = Path(sys.argv[1])
        if not INPUT.exists():
            print("File not found:", INPUT); sys.exit(1)
    else:
        choice = choose_file(DATA_FOLDER)
        if choice is None:
            sys.exit(0)
        INPUT = choice

    print("Parsing file:", INPUT)
    t, gx_deg, gy_deg, gz_deg = parse_log(INPUT)
    if t.size < 3:
        print("Not enough timestamped samples (need >=3)."); sys.exit(1)

    # sort and timing diagnostics for raw timestamps
    order = np.argsort(t)
    t = t[order]
    gx_deg = gx_deg[order]; gy_deg = gy_deg[order]; gz_deg = gz_deg[order]
    dt = np.diff(t)
    print(f"Raw samples: {t.size}, raw duration = {t[-1]-t[0]:.2f} s")
    print(f"Raw dt: mean={np.mean(dt):.6f}s  std={np.std(dt):.6f}s  min={np.min(dt):.6f}s  max={np.max(dt):.6f}s")

    # convert deg/s -> rad/s BEFORE resampling / Allan computation
    deg2rad = math.pi / 180.0
    gx = gx_deg * deg2rad
    gy = gy_deg * deg2rad
    gz = gz_deg * deg2rad

    # resample to uniform grid using median dt (but compute raw duration separately)
    try:
        t_u, gx_u, fs, mean_dt, std_dt, median_dt = resample_uniform(t, gx, target_fs=None)
    except Exception as e:
        print("Resampling error:", e); sys.exit(1)
    gy_u = np.interp(t_u, t, gy)
    gz_u = np.interp(t_u, t, gz)
    N_uniform = t_u.size
    resampled_total_time = t_u[-1] - t_u[0]
    raw_total_time = t[-1] - t[0]
    print(f"Resampled to uniform grid: N={N_uniform}, fs≈{fs:.4f} Hz, resampled_total_time={resampled_total_time:.2f}s")
    print(f"Raw total_time (from timestamps): {raw_total_time:.2f} s")

    # Determine compute limit (based on raw timestamps to avoid resampling shrinkage)
    tau_max_compute = min(TAU_MAX_REQ, int(math.floor(raw_total_time)))
    if tau_max_compute < 1:
        print("Recording too short for tau >= 1s."); sys.exit(1)

    # Display axis will always be 1..TAU_MAX_REQ (one tick per second as requested)
    taus_display = np.arange(1, TAU_MAX_REQ + 1, 1)
    taus_compute = np.arange(1, tau_max_compute + 1, 1)  # integer seconds we will compute

    # prepare full arrays for display (NaN where not computed)
    sigma2_x_full = np.full(taus_display.shape, np.nan)
    sigma2_y_full = np.full(taus_display.shape, np.nan)
    sigma2_z_full = np.full(taus_display.shape, np.nan)
    K_used_full    = np.zeros(taus_display.shape, dtype=int)

    # compute Allan variance only where data permits (1..tau_max_compute)
    for tau in taus_compute:
        idx = tau - 1
        s2x, kx = allan_variance_overlapping_from_uniform(gx_u, fs, float(tau))
        s2y, ky = allan_variance_overlapping_from_uniform(gy_u, fs, float(tau))
        s2z, kz = allan_variance_overlapping_from_uniform(gz_u, fs, float(tau))
        if s2x is not None and s2x > 0.0:
            sigma2_x_full[idx] = s2x
        if s2y is not None and s2y > 0.0:
            sigma2_y_full[idx] = s2y
        if s2z is not None and s2z > 0.0:
            sigma2_z_full[idx] = s2z
        # conservative K: min positive K among axes computed for that tau
        k_vals = [k for k in (kx, ky, kz) if k is not None and k > 0]
        K_used_full[idx] = min(k_vals) if k_vals else 0

    # Save numeric results (sigma^2 in rad^2 / s^2)
    out_txt = "allan_variance_rad2_full.txt"
    with open(out_txt, "w") as fo:
        fo.write("#tau_s\tsigma2_x\tsigma2_y\tsigma2_z\tK_used\n")
        for tau, s2x, s2y, s2z, K in zip(taus_display, sigma2_x_full, sigma2_y_full, sigma2_z_full, K_used_full):
            fo.write(f"{int(tau)}\t{(s2x if not np.isnan(s2x) else 0.0):.12e}\t{(s2y if not np.isnan(s2y) else 0.0):.12e}\t{(s2z if not np.isnan(s2z) else 0.0):.12e}\t{int(K)}\n")
    print("Saved numeric results to", out_txt)

    # ------------------ Plot Allan VARIANCE (sigma^2) in rad^2 / s^2 ------------------
    plt.figure(figsize=(14,6))

    # plot one point per integer tau (NaNs shown as gaps)
    plt.plot(taus_display, sigma2_x_full, marker='o', markersize=3, linestyle='-', linewidth=0.7, label=r'$\sigma_x^2$ (rad$^2$/s$^2$)')
    plt.plot(taus_display, sigma2_y_full, marker='s', markersize=3, linestyle='-', linewidth=0.7, label=r'$\sigma_y^2$ (rad$^2$/s$^2$)')
    plt.plot(taus_display, sigma2_z_full, marker='^', markersize=3, linestyle='-', linewidth=0.7, label=r'$\sigma_z^2$ (rad$^2$/s$^2$)')

    # mark unreliable taus (small K)
    min_K_trust = 8
    unreliable_mask = (K_used_full > 0) & (K_used_full < min_K_trust)
    if unreliable_mask.any():
        # use hollow grey circles plotted at the first non-NaN value among axes for visibility
        # build a y vector to scatter where at least one axis is present
        y_for_scatter = np.full(taus_display.shape, np.nan)
        present_any = (~np.isnan(sigma2_x_full)) | (~np.isnan(sigma2_y_full)) | (~np.isnan(sigma2_z_full))
        y_for_scatter[present_any] = np.nanmax(np.vstack([
            np.nan_to_num(sigma2_x_full, nan=-np.inf),
            np.nan_to_num(sigma2_y_full, nan=-np.inf),
            np.nan_to_num(sigma2_z_full, nan=-np.inf)
        ]), axis=0)[present_any]
        plt.scatter(taus_display[unreliable_mask], y_for_scatter[unreliable_mask],
                    facecolors='none', edgecolors='gray', s=30, label=f'K<{min_K_trust} (unreliable)')

    plt.xlabel('Tau (s)')
    plt.ylabel('Allan variance σ² (rad$^2$/s$^2$)')
    plt.title(f'Allan variance vs τ — linear axes (display 1..{TAU_MAX_REQ}), one tick per second')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend(loc='upper right')

    # force x-limits and ONE tick per second (user requested)
    plt.xlim(1, TAU_MAX_REQ)
    plt.xticks(np.arange(1, TAU_MAX_REQ + 1, 1))

    # Secondary axis: show K_used to explain reliability of points
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(taus_display, K_used_full, linestyle='--', linewidth=0.6, color='0.3', label='K_used (overlaps)')
    ax2.set_ylabel('K (number of overlapping differences)')
    kmax = int(np.nanmax(K_used_full)) if np.any(K_used_full) else 10
    ax2.set_ylim(0, max(10, kmax + 2))
    ax2.legend(loc='lower right')

    # mark recommended T_init (using variance arrays directly)
    tx = choose_Tinit_from_sigma(taus_display, sigma2_x_full, K_used_full, tol_rel=1.10, min_K=min_K_trust)
    ty = choose_Tinit_from_sigma(taus_display, sigma2_y_full, K_used_full, tol_rel=1.10, min_K=min_K_trust)
    tz = choose_Tinit_from_sigma(taus_display, sigma2_z_full, K_used_full, tol_rel=1.10, min_K=min_K_trust)
    print("Per-axis T_init candidates (tau where sigma^2 within 10% of min and K>=min_K):", tx, ty, tz)
    cands = [v for v in (tx, ty, tz) if v is not None]
    if cands:
        T_init_reco = max(cands)
        print("Recommended conservative T_init:", T_init_reco, "s")
        plt.axvline(T_init_reco, color='k', linestyle='--', linewidth=1)
        ymin, ymax = ax.get_ylim()
        plt.text(T_init_reco * 1.01, ymin + 0.03 * (ymax - ymin), f"T_init={T_init_reco}s", rotation=90, va='bottom')

    plt.tight_layout()
    png = "allan_variance_rad2_full_linear.png"
    plt.savefig(png, dpi=300)
    print("Saved plot (variance) to", png)
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed (probably headless). PNG saved instead. Error:", e)
    print("Done.")