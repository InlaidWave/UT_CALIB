import argparse
import re
from pathlib import Path
from typing import Tuple, List

import numpy as np

# Patterns to parse gyro lines like: &GX <val> GY <val> GZ <val>
GYRO_PATTERN = re.compile(r"&?GX\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*GY\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*GZ\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def load_gyro_series(file_path: Path) -> np.ndarray:
    """
    Load all gyro samples (GX,GY,GZ) from a log file, ignoring segments.
    Returns array (N,3). If none found, returns empty array.
    """
    vals: List[List[float]] = []
    with file_path.open('r') as f:
        for raw in f:
            m = GYRO_PATTERN.search(raw)
            if m:
                gx, gy, gz = map(float, m.groups())
                vals.append([gx, gy, gz])
    return np.array(vals, dtype=float)


def allan_variance(series: np.ndarray, rate_hz: float, t0: float = 1.0, tn: float = None, num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Allan variance for each axis in series (N,3) over a set of tau values.
    Returns (taus, sigma2) where sigma2 has shape (len(taus), 3).
    """
    if series.ndim != 2 or series.shape[1] != 3:
        raise ValueError("series must be (N,3)")
    n = len(series)
    if n < 2:
        return np.array([]), np.zeros((0, 3))

    total_time = n / rate_hz
    if tn is None:
        tn = max(t0 * 2, total_time / 2)
    tn = min(tn, total_time)

    taus = np.geomspace(max(t0, 1.0 / rate_hz), tn, num_points)
    sigma2 = np.zeros((len(taus), 3))

    # Precompute cumulative sums for fast interval averages
    csum = np.cumsum(series, axis=0)
    csum = np.vstack([np.zeros((1, 3)), csum])  # pad

    for i, tau in enumerate(taus):
        m = int(round(tau * rate_hz))
        if m < 1:
            sigma2[i, :] = np.nan
            continue
        # number of non-overlapping intervals of length m
        K = (n // m) - 1
        if K <= 0:
            sigma2[i, :] = np.nan
            continue
        # Compute interval averages x_k over length m
        # x_k = (1/m) * sum( x[km : (k+1)m - 1] )
        xk = (csum[m::m] - csum[:-m:m]) / m  # shape (n//m, 3)
        # First differences squared
        diffs = xk[1:K+1] - xk[:K]
        sigma2[i, :] = 0.5 * np.mean(diffs**2, axis=0)

    return taus, sigma2


def recommend_tinit(taus: np.ndarray, sigma2: np.ndarray, slope_thresh: float = 0.1, window: int = 3) -> float:
    """
    Heuristic: choose the smallest tau where all axes exhibit near-plateau behavior
    (|d log sigma / d log tau| < slope_thresh) for 'window' consecutive points.
    Returns tau if found, else fallback to median tau.
    """
    if taus.size == 0:
        return 10.0
    # Use log-log slope
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau = np.log(taus)
        slopes = []
        for ax in range(3):
            s = np.gradient(np.log(np.sqrt(sigma2[:, ax])), log_tau)
            slopes.append(s)
        slopes = np.stack(slopes, axis=1)  # (len(taus), 3)
        good = np.all(np.abs(slopes) < slope_thresh, axis=1)
        # Check for consecutive window of 'good'
        cnt = 0
        for i, ok in enumerate(good):
            cnt = cnt + 1 if ok else 0
            if cnt >= window:
                return float(taus[i - window + 1])
    return float(np.median(taus))


def main():
    ap = argparse.ArgumentParser(description="Compute Allan variance and recommend Tinit from a log with GX/GY/GZ lines.")
    ap.add_argument("logfile", type=str, help="Path to log file (e.g., DATA/calib_data_*.txt)")
    ap.add_argument("--rate", type=float, default=200.0, help="Sampling rate (Hz) of gyro samples")
    ap.add_argument("--units", type=str, default="deg_s", choices=["deg_s", "rad_s"], help="Units of GX/GY/GZ values")
    ap.add_argument("--t0", type=float, default=1.0, help="Min tau (s)")
    ap.add_argument("--tn", type=float, default=None, help="Max tau (s), default half of series duration")
    args = ap.parse_args()

    path = Path(args.logfile)
    series = load_gyro_series(path)
    if series.size == 0:
        print("No gyro samples found in the file.")
        return

    # Convert to rad/s if needed
    if args.units.lower() == "deg_s":
        series = np.deg2rad(series)

    taus, sigma2 = allan_variance(series, rate_hz=args.rate, t0=args.t0, tn=args.tn)
    if taus.size == 0:
        print("Not enough data to compute Allan variance.")
        return

    tinit = recommend_tinit(taus, sigma2)
    print(f"Recommended Tinit (s): {tinit:.2f}")

    # Bias estimate from first Tinit seconds
    n0 = int(round(tinit * args.rate))
    n0 = max(1, min(n0, len(series)))
    bias = np.mean(series[:n0, :], axis=0)
    if args.units.lower() == "deg_s":
        # Report also in deg/s for convenience
        bias_deg = np.rad2deg(bias)
        print(f"Bias (rad/s): {bias}")
        print(f"Bias (deg/s): {bias_deg}")
    else:
        print(f"Bias (rad/s): {bias}")


if __name__ == "__main__":
    main()
