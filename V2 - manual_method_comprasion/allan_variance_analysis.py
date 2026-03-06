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
 - TAU_MAX_REQ defines the displayed x-axis range (default 1800 s).
 - The script computes Allan variance only where raw timestamps give enough
   data; remaining taus are left NaN (gaps in plot). The x-axis still runs
   to TAU_MAX_REQ but the major/minor tick strategy avoids plotting thousands
   of heavy major ticks which can freeze matplotlib in large ranges.
"""
import sys, re, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_FOLDER = "SERVO_DATA"
TAU_MAX_REQ = 1800  # display 1..TAU_MAX_REQ seconds on x-axis

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

# ---------------- fast overlapping Allan variance (cumsum) ----------------
def allan_variance_overlapping_from_uniform_cumsum(x_uniform, fs, tau_s, cumsum_x=None):
    """
    Fast overlapping Allan variance using cumulative sums for moving averages.

    x_uniform : evenly sampled series (rad/s)
    fs        : sampling frequency (Hz)
    tau_s     : integration time in seconds (float)
    cumsum_x  : optional precomputed cumulative sum array with leading zero:
                cumsum_x = np.concatenate(([0.0], np.cumsum(x_uniform)))
    Returns (sigma2, K) where K = N - 2*m + 1 (number of overlapping differences),
    or (None, 0) if not enough points.
    """
    if tau_s <= 0:
        return None, 0
    m = int(round(tau_s * fs))
    if m < 1:
        m = 1
    x = x_uniform
    N = x.size
    if 2 * m >= N:
        return None, 0

    if cumsum_x is None:
        cumsum_x = np.concatenate(([0.0], np.cumsum(x)))

    # moving average mu of window length m:
    # mu[i] = (cumsum_x[i+m] - cumsum_x[i]) / m   for i=0..N-m
    mu = (cumsum_x[m:] - cumsum_x[:-m]) / float(m)

    # differences separated by m: diffs = mu[m:] - mu[:-m]
    if mu.size <= m:
        return None, 0
    diffs = mu[m:] - mu[:-m]
    K = diffs.size
    if K <= 0:
        return None, 0
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

def effective_tau_limit_from_N(N, fs, min_K=8):
    """
    Return the largest integer tau (in seconds) such that K = N - 2*m + 1 >= min_K,
    with m = round(tau * fs). If even tau=1 fails, returns 0.
    """
    if N <= 0:
        return 0
    # Solve for m: N - 2*m + 1 >= min_K  => m <= (N - min_K + 1) / 2
    m_max = int(np.floor((N - min_K + 1) / 2.0))
    if m_max < 1:
        return 0
    tau_max = int(np.floor(m_max / float(fs)))
    return tau_max

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
    tau_max_by_time = int(math.floor(raw_total_time))
    if tau_max_by_time < 1:
        print("Recording too short for tau >= 1s."); sys.exit(1)

    # prepare display taus 1..TAU_MAX_REQ
    taus_display = np.arange(1, TAU_MAX_REQ + 1, 1)

    # precompute cumsums once for speed
    cumsum_x = np.concatenate(([0.0], np.cumsum(gx_u)))
    cumsum_y = np.concatenate(([0.0], np.cumsum(gy_u)))
    cumsum_z = np.concatenate(([0.0], np.cumsum(gz_u)))

    # decide minimum K required for trust plotting/compute
    min_K_trust = 8

    # conservative effective compute limit to ensure K >= min_K_trust for each axis
    tau_max_by_K_x = effective_tau_limit_from_N(N_uniform, fs, min_K=min_K_trust)
    tau_max_by_K_y = effective_tau_limit_from_N(N_uniform, fs, min_K=min_K_trust)
    tau_max_by_K_z = effective_tau_limit_from_N(N_uniform, fs, min_K=min_K_trust)
    # take MIN across axes to be conservative (ensures all axes have >= min_K)
    tau_max_by_K = min(tau_max_by_K_x, tau_max_by_K_y, tau_max_by_K_z)
    if tau_max_by_K < 1:
        # fallback: allow at least tau=1 to be computed if possible
        tau_max_by_K = 1

    # final compute limit: do not compute beyond raw-duration nor TAU_MAX_REQ nor K-limit
    tau_max_compute = min(TAU_MAX_REQ, tau_max_by_time, tau_max_by_K)
    print(f"Computing taus 1..{tau_max_compute} (limited by raw duration and K>={min_K_trust} across axes)")

    taus_compute = np.arange(1, tau_max_compute + 1, 1)  # integer seconds we will compute

    # prepare full arrays for display (NaN where not computed)
    sigma2_x_full = np.full(taus_display.shape, np.nan)
    sigma2_y_full = np.full(taus_display.shape, np.nan)
    sigma2_z_full = np.full(taus_display.shape, np.nan)
    K_used_full    = np.zeros(taus_display.shape, dtype=int)

    # compute Allan variance only where data permits (1..tau_max_compute)
    for tau in taus_compute:
        idx = tau - 1
        s2x, kx = allan_variance_overlapping_from_uniform_cumsum(gx_u, fs, float(tau), cumsum_x=cumsum_x)
        s2y, ky = allan_variance_overlapping_from_uniform_cumsum(gy_u, fs, float(tau), cumsum_x=cumsum_y)
        s2z, kz = allan_variance_overlapping_from_uniform_cumsum(gz_u, fs, float(tau), cumsum_x=cumsum_z)
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
    fig, ax = plt.subplots(figsize=(14,6))

    # plot one point per integer tau (NaNs shown as gaps)
    ax.plot(taus_display, sigma2_x_full, marker='o', markersize=3, linestyle='-', linewidth=0.7, label=r'$\sigma_x^2$ (rad$^2$/s$^2$)')
    ax.plot(taus_display, sigma2_y_full, marker='s', markersize=3, linestyle='-', linewidth=0.7, label=r'$\sigma_y^2$ (rad$^2$/s$^2$)')
    ax.plot(taus_display, sigma2_z_full, marker='^', markersize=3, linestyle='-', linewidth=0.7, label=r'$\sigma_z^2$ (rad$^2$/s$^2$)')

    # mark unreliable taus (small K)
    unreliable_mask = (K_used_full > 0) & (K_used_full < min_K_trust)
    if unreliable_mask.any():
        # use hollow grey circles plotted at the first non-NaN value among axes for visibility
        y_for_scatter = np.full(taus_display.shape, np.nan)
        present_any = (~np.isnan(sigma2_x_full)) | (~np.isnan(sigma2_y_full)) | (~np.isnan(sigma2_z_full))
        # build a per-tau max across axes for placement
        stacked = np.vstack([
            np.nan_to_num(sigma2_x_full, nan=-np.inf),
            np.nan_to_num(sigma2_y_full, nan=-np.inf),
            np.nan_to_num(sigma2_z_full, nan=-np.inf)
        ])
        y_for_scatter[present_any] = np.nanmax(stacked, axis=0)[present_any]
        ax.scatter(taus_display[unreliable_mask], y_for_scatter[unreliable_mask],
                   facecolors='none', edgecolors='gray', s=30, label=f'K<{min_K_trust} (unreliable)')

    ax.set_xlabel('Tau (s)')
    ax.set_ylabel('Allan variance σ² (rad$^2$/s$^2$)')
    ax.set_title(f'Allan variance vs τ — linear axes (display 1..{TAU_MAX_REQ}), condensed ticks for large range')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(loc='upper right')

    # x-limits fixed
    ax.set_xlim(1, TAU_MAX_REQ)

    # Tick strategy:
    # - If display length is small, put a major tick each second (user request preserved when reasonable).
    # - For large TAU_MAX_REQ, use major ticks every 10 s and minor ticks every 1 s so visually there's a 1s grid
    display_len = TAU_MAX_REQ
    if display_len <= 600:
        major_locator = mticker.MultipleLocator(1)
        minor_locator = mticker.AutoMinorLocator(1)
    else:
        major_locator = mticker.MultipleLocator(10)
        minor_locator = mticker.MultipleLocator(1)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.tick_params(axis='x', which='minor', length=4, direction='out')

    # Secondary axis: show K_used to explain reliability of points
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
        ax.axvline(T_init_reco, color='k', linestyle='--', linewidth=1)
        ymin, ymax = ax.get_ylim()
        ax.text(T_init_reco * 1.01, ymin + 0.03 * (ymax - ymin), f"T_init={T_init_reco}s", rotation=90, va='bottom')

    plt.tight_layout()
    png = "allan_variance_rad2_full_linear.png"
    plt.savefig(png, dpi=300)
    print("Saved plot (variance) to", png)
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed (probably headless). PNG saved instead. Error:", e)
    print("Done.")