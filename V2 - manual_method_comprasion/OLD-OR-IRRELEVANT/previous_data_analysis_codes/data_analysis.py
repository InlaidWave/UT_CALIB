import os
import re
import sys
import argparse
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

# =========================
# Configuration
# =========================
data_folder = "DATA"
g = 9.81872  # local gravity magnitude (m/s^2)

# Gyro assumptions (adjust if needed)
SAMPLE_RATE_HZ = 200.0  # IMU gyro sampling rate during motion segments
GYRO_UNITS = "deg_s"    # incoming units in the log: 'deg_s' or 'rad_s'

# Gyro calibration mode:
# - 'fit_bias': fit bias + scales (+ misalignment) from motion segments (your current preference)
# - 'bias_from_static': subtract bias measured in an initial static period (per paper), fit scales (+ misalignment)
GYRO_CALIB_MODE = "fit_bias"

# Include gyro misalignment Tg in the fit (6 parameters). If False, Tg=I.
GYRO_FIT_MISALIGNMENT = True
ACCEL_UNITS = "g"  # 'g' (unit gravity) or 'ms2' (m/s^2)


# =========================
# Utilities: listing files
# =========================
def list_txt_files(directory):
    folder_path = os.path.join(directory, data_folder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]


# =========================
# Parsing logs (accel poses + gyro segments)
# =========================
accel_pattern = re.compile(r"&?X\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*Y\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*Z\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
gyro_pattern = re.compile(r"&?GX\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*GY\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*GZ\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
mode_start_pattern = re.compile(r"&?MODE\s*=\s*GYRO_START", re.IGNORECASE)
mode_end_pattern = re.compile(r"&?MODE\s*=\s*GYRO_END", re.IGNORECASE)


def parse_log(file_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns:
      accel_poses: (M,3) accelerometer averaged vectors (raw units assumed g=1 scaled; will be scaled to m/s^2 later)
      gyro_segments: list of arrays, each (N,3) of gyro samples for motion between poses k-1 -> k
    """
    accel_poses: List[List[float]] = []
    gyro_segments: List[np.ndarray] = []
    current_segment: List[List[float]] = []
    in_segment = False

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Mode markers for gyro segments
            if mode_start_pattern.search(line):
                if in_segment and current_segment:
                    # close previous just in case
                    gyro_segments.append(np.array(current_segment, dtype=float))
                    current_segment = []
                in_segment = True
                continue
            if mode_end_pattern.search(line):
                if in_segment:
                    gyro_segments.append(np.array(current_segment, dtype=float))
                    current_segment = []
                    in_segment = False
                continue

            # Gyro samples (only valid inside segments ideally)
            m_g = gyro_pattern.search(line)
            if m_g:
                gx, gy, gz = map(float, m_g.groups())
                if in_segment:
                    current_segment.append([gx, gy, gz])
                # if outside a segment, ignore or buffer – we'll ignore to enforce clean segments
                continue

            # Accel pose samples (one per static pose)
            m_a = accel_pattern.search(line)
            if m_a:
                ax, ay, az = map(float, m_a.groups())
                accel_poses.append([ax, ay, az])
                continue

            # ignore other lines

    # finalize if file ended inside a segment
    if in_segment and current_segment:
        gyro_segments.append(np.array(current_segment, dtype=float))

    accel_poses = np.array(accel_poses, dtype=float)
    return accel_poses, gyro_segments


# =========================
# Accelerometer calibration (sphere fit)
# =========================
def build_Ta(alpha_yz, alpha_zy, alpha_zx):
    """Upper-triangular Ta as in paper Eq. (2): uses three angles (assuming BF≡AOF)."""
    return np.array([
        [1.0,        -alpha_yz,  alpha_zy],
        [0.0,         1.0,      -alpha_zx],
        [0.0,         0.0,       1.0],
    ])


def build_K(sx, sy, sz):
    return np.diag([sx, sy, sz])


def accel_residuals(params, data_ms2):
    """
    params: [bx, by, bz, sx, sy, sz, alpha_yz, alpha_zy, alpha_zx]
    data_ms2: (N,3) raw accelerometer vectors in m/s^2 (not calibrated)
    """
    bx, by, bz, sx, sy, sz, a_yz, a_zy, a_zx = params
    b = np.array([bx, by, bz])
    K = build_K(sx, sy, sz)
    T = build_Ta(a_yz, a_zy, a_zx)
    # residual: ||Ta Ka (a + b)|| - g
    a_corr = (T @ (K @ (data_ms2.T + b[:, None]))).T  # (N,3)
    norms = np.linalg.norm(a_corr, axis=1)
    return norms - g


def accel_find_params(data_ms2: np.ndarray) -> np.ndarray:
    init = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    res = least_squares(accel_residuals, init, args=(data_ms2,))
    return res.x


def accel_apply(data_ms2: np.ndarray, params: np.ndarray) -> np.ndarray:
    bx, by, bz, sx, sy, sz, a_yz, a_zy, a_zx = params
    b = np.array([bx, by, bz])
    K = build_K(sx, sy, sz)
    T = build_Ta(a_yz, a_zy, a_zx)
    return (T @ (K @ (data_ms2.T + b[:, None]))).T


def accel_summary(calib_ms2: np.ndarray, raw_ms2: np.ndarray):
    def pct_mean_error(residuals, percentile=95):
        cutoff = np.percentile(np.abs(residuals), percentile)
        top = np.abs(residuals)[np.abs(residuals) >= cutoff]
        return float(np.mean(top)) if top.size else 0.0

    m1 = np.linalg.norm(calib_ms2, axis=1)
    m2 = np.linalg.norm(raw_ms2, axis=1)
    r1 = m1 - g
    r2 = m2 - g
    print(f"Accel: mean offset from g (calib): {np.mean(r1):.6f}")
    print(f"Accel: mean offset from g (raw):   {np.mean(r2):.6f}")
    print(f"Accel: 95th%% mean abs error (calib): {pct_mean_error(r1):.6f}")
    print(f"Accel: 95th%% mean abs error (raw):   {pct_mean_error(r2):.6f}")
    print(f"Accel: RMS fit to sphere (calib): {np.sqrt(np.mean(r1**2)):.6f}")
    print(f"Accel: RMS fit to sphere (raw):   {np.sqrt(np.mean(r2**2)):.6f}")



# =========================
# Gyroscope calibration (quaternion RK4n, versor cost)
# =========================
def build_Tg(g_yz, g_zy, g_xz, g_zx, g_xy, g_yx):
    return np.array([
        [1.0,    -g_yz,  g_zy],
        [g_xz,    1.0,  -g_zx],
        [-g_xy,   g_yx,  1.0],
    ])


def omega_to_rad_s(omega):
    if GYRO_UNITS.lower() == "deg_s":
        return np.deg2rad(omega)
    return omega  # assume rad/s


def quat_omega_matrix(omega):
    """Return 4x4 Omega(omega) matrix for quaternion kinematics."""
    wx, wy, wz = omega
    return np.array([
        [0.0,   -wx,   -wy,   -wz],
        [wx,    0.0,    wz,   -wy],
        [wy,   -wz,    0.0,    wx],
        [wz,    wy,   -wx,    0.0],
    ])


def quat_mul(q, r):
    """Hamilton product q*r for quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def rk4n_integrate_quat(omegas_rad_s: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate quaternion from identity over a sequence of angular rates.
    omegas_rad_s: (N,3) in rad/s
    Returns: final quaternion [w,x,y,z]
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    for w in omegas_rad_s:
        # f(q,t) = 0.5 * Omega(omega) * q
        def f(qv):
            return 0.5 * (quat_omega_matrix(w) @ qv)

        k1 = f(q)
        k2 = f(q + 0.5*dt*k1)
        k3 = f(q + 0.5*dt*k2)
        k4 = f(q + dt*k3)
        q = q + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        q = quat_normalize(q)
    return q


def quat_rotate(q, v):
    """Rotate vector v by quaternion q ([w,x,y,z])."""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])
    return R @ v


def build_gyro_residuals(accel_versors: np.ndarray,
                         gyro_segments: List[np.ndarray],
                         params: np.ndarray) -> np.ndarray:
    """
    Simplified, fast residual builder using Rodrigues' rotation formula instead of RK4 quaternions.
    params packs (order depends on flags):
      if GYRO_FIT_MISALIGNMENT: [g_yz,g_zy,g_xz,g_zx,g_xy,g_yx, sgx,sgy,sgz, (optional) bgx,bgy,bgz]
      else:                       [sgx,sgy,sgz, (optional) bgx,bgy,bgz]
    """
    idx = 0
    if GYRO_FIT_MISALIGNMENT:
        g_yz, g_zy, g_xz, g_zx, g_xy, g_yx = params[idx:idx+6]; idx += 6
        Tg = build_Tg(g_yz, g_zy, g_xz, g_zx, g_xy, g_yx)
    else:
        Tg = np.eye(3)

    sgx, sgy, sgz = params[idx:idx+3]; idx += 3
    Kg = build_K(sgx, sgy, sgz)

    if idx < len(params):
        bg = np.array(params[idx:idx+3])
    else:
        bg = np.zeros(3)

    dt = 1.0 / SAMPLE_RATE_HZ
    res = []

    for k in range(1, len(accel_versors)):
        ua_prev = accel_versors[k-1]
        ua_k = accel_versors[k]
        seg = gyro_segments[k-1]
        if seg.size == 0:
            continue

        # Apply calibration to gyro: ω^O = Tg·Kg·(ω^S + b)
        omega_S = seg
        omega_cal = (Tg @ (Kg @ (omega_S.T + bg[:, None]))).T
        omega_rad = omega_to_rad_s(omega_cal)

        # --- FAST INTEGRATION (Rodrigues rotation) ---
        delta_theta = np.sum(omega_rad, axis=0) * dt
        theta = np.linalg.norm(delta_theta)
        if theta < 1e-12:
            R = np.eye(3)
        else:
            k_axis = delta_theta / theta
            K = np.array([[0, -k_axis[2], k_axis[1]],
                          [k_axis[2], 0, -k_axis[0]],
                          [-k_axis[1], k_axis[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        ug_k = R @ ua_prev
        res.append(ug_k - ua_k)

    return np.concatenate(res) if res else np.zeros(0)

def calibrate_gyro(accel_ms2_calib: np.ndarray,
                   gyro_segments: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
    # Build accel gravity versors from calibrated accel poses
    accel_versors = accel_ms2_calib / np.linalg.norm(accel_ms2_calib, axis=1, keepdims=True)

    # Initial parameters
    p = []
    if GYRO_FIT_MISALIGNMENT:
        p += [0, 0, 0, 0, 0, 0]  # gammas
    p += [1.0, 1.0, 1.0]  # scales
    if GYRO_CALIB_MODE == "fit_bias":
        p += [0.0, 0.0, 0.0]  # bias
    p = np.array(p, dtype=float)

    def fun(params):
        return build_gyro_residuals(accel_versors, gyro_segments, params)

    # ---- NEW: safer, bounded least-squares ----
    # Verbose=2 to show iteration progress
    # max_nfev=1000 caps computation time
    # loss='soft_l1' makes it more robust to outliers
    print("Starting gyro optimization (this may take ~10–30 s)...")

    if GYRO_FIT_MISALIGNMENT:
        gamma_bound = 0.1
        lower = [-gamma_bound]*6 + [0.8, 0.8, 0.8, -20, -20, -20]
        upper = [ gamma_bound]*6 + [1.2, 1.2, 1.2,  20,  20,  20]
    else:
        lower = [0.8, 0.8, 0.8, -20, -20, -20]
        upper = [1.2, 1.2, 1.2,  20,  20,  20]

    res = least_squares(fun, p,
                        bounds=(lower, upper),
                        max_nfev=300,
                        verbose=2,
                        loss='soft_l1',
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10)

    # Unpack results
    out = {}
    idx = 0
    if GYRO_FIT_MISALIGNMENT:
        out["gamma"] = res.x[idx:idx+6]; idx += 6
    out["scale"] = res.x[idx:idx+3]; idx += 3
    if GYRO_CALIB_MODE == "fit_bias":
        out["bias"] = res.x[idx:idx+3]
    else:
        out["bias"] = np.zeros(3)
    out["cost"] = 0.5 * np.sum(res.fun**2)
    out["success"] = res.success
    return res.x, out


# =========================
# Main driver
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Accelerometer + Gyroscope calibration from logged data.")
    ap.add_argument("--file", type=str, default=None, help="Path to a DATA/*.txt log; if omitted, you will be prompted (or auto-picked if only one).")
    ap.add_argument("--rate", type=float, default=None, help="Gyro sample rate (Hz), overrides module default.")
    ap.add_argument("--units", type=str, choices=["deg_s", "rad_s"], default=None, help="Gyro units, overrides module default.")
    ap.add_argument("--mode", type=str, choices=["fit_bias", "bias_from_static"], default=None, help="Gyro bias handling mode.")
    ap.add_argument("--no-misalign", action="store_true", help="Disable fitting gyro misalignment (Tg=I).")
    ap.add_argument("--accel-units", type=str, choices=["g", "ms2"], default=None, help="Accelerometer units in the log (default: auto-detect, fallback to 'g').")
    args = ap.parse_args()

    # Apply overrides (set module-level constants via globals dict)
    if args.rate is not None:
        globals()['SAMPLE_RATE_HZ'] = args.rate
    if args.units is not None:
        globals()['GYRO_UNITS'] = args.units
    if args.mode is not None:
        globals()['GYRO_CALIB_MODE'] = args.mode
    if args.no_misalign:
        globals()['GYRO_FIT_MISALIGNMENT'] = False
    if args.accel_units is not None:
        globals()['ACCEL_UNITS'] = args.accel_units

    # Resolve file path
    data_file = None
    if args.file:
        data_file = args.file
    else:
        txt_files = list_txt_files(".")
        if not txt_files:
            print("No .txt files found under DATA/")
            sys.exit(1)
        if len(txt_files) == 1:
            data_file = os.path.join(data_folder, txt_files[0])
            print(f"Using only file found: {txt_files[0]}")
        else:
            # Non-interactive environments might not like input(); auto-pick newest as fallback
            try:
                print("Available .txt files:")
                for idx, fname in enumerate(txt_files):
                    print(f"{idx}: {fname}")
                choice = int(input("Select a file by number: "))
                data_file = os.path.join(data_folder, txt_files[choice])
            except Exception:
                # Pick the lexicographically last (often newest)
                newest = sorted(txt_files)[-1]
                data_file = os.path.join(data_folder, newest)
                print(f"No input; auto-selecting latest file: {newest}")

    # Parse file: accel poses and gyro segments
    accel_raw, gyro_segments = parse_log(data_file)
    if accel_raw.size == 0:
        print("No accelerometer pose samples found. Exiting.")
        sys.exit(1)

    # Accelerometer units: auto-detect if not overridden
    accel_units = ACCEL_UNITS
    if args.accel_units is None:
        # crude heuristic: mean norm near 1 => 'g'; near g => 'ms2'
        norms = np.linalg.norm(accel_raw, axis=1) if accel_raw.size else np.array([0.0])
        mean_norm = float(np.mean(norms)) if norms.size else 0.0
        if 0.3 <= mean_norm <= 2.0:
            accel_units = "g"
        elif 3.0 <= mean_norm <= 20.0:
            accel_units = "ms2"
        else:
            accel_units = "g"  # fallback
    if accel_units == "g":
        accel_ms2 = accel_raw * g
    else:
        accel_ms2 = accel_raw

    # Calibrate accelerometer (sphere fit)
    acc_params = accel_find_params(accel_ms2)
    accel_ms2_calib = accel_apply(accel_ms2, acc_params)
    print("\nAccelerometer calibration parameters:")
    print(f"  Bias (bx,by,bz): {acc_params[:3]}")
    print(f"  Scales (sx,sy,sz): {acc_params[3:6]}")
    print(f"  Misalign (alpha_yz,alpha_zy,alpha_zx): {acc_params[6:]}")
    print("")
    accel_summary(accel_ms2_calib, accel_ms2)

    # Gyro calibration if segments present
    if gyro_segments:
        print(f"\nFound {len(gyro_segments)} gyro motion segments.")

        # Sanity check, segments should be len(accel)-1; if mismatch, cut to fit
        if len(gyro_segments) != len(accel_ms2_calib) - 1:
            n = min(len(gyro_segments), len(accel_ms2_calib) - 1)
            gyro_segments = gyro_segments[:n]
            accel_ms2_calib = accel_ms2_calib[: n + 1]
            print(f"Adjusted to {n} segments to match accel poses.")

        # ---- NEW: optional limit to avoid heavy optimization ----
        if len(gyro_segments) > 12:
            print(f"Limiting to first 12 segments for faster calibration (out of {len(gyro_segments)})")
            gyro_segments = gyro_segments[:12]
            accel_ms2_calib = accel_ms2_calib[:13]

        _, gyro_info = calibrate_gyro(accel_ms2_calib, gyro_segments)
        print("\nGyroscope calibration results:")
        if GYRO_FIT_MISALIGNMENT:
            gamma = gyro_info["gamma"]
            print(f"  Misalign gamma (yz, zy, xz, zx, xy, yx): {gamma}")
        scale = gyro_info["scale"]
        print(f"  Scales (sgx,sgy,sgz): {scale}")
        bias = gyro_info["bias"]
        print(f"  Bias (bgx,bgy,bgz): {bias}")
        print(f"  Success: {gyro_info['success']}  Cost: {gyro_info['cost']:.6f}")
    else:
        print("\nNo gyro segments found in the log. Skipping gyro calibration.")