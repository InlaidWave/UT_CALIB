# fixed_calib.py
import os, re, sys, argparse
import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================
# Configuration
# =========================
data_folder = "DATA"
g = 9.81872
SAMPLE_RATE_HZ = 70.0
GYRO_UNITS = "rad_s"
GYRO_AXIS_SIGN = np.array([1.0, 1.0, 1.0])
GYRO_CALIB_MODE = "fit_bias"   # or "use_static"
GYRO_FIT_MISALIGNMENT = False
ACCEL_UNITS = "g"
GYRO_INTEGRATOR = "rk4"

# =========================
# Helpers
# =========================
def list_txt_files(directory):
    folder_path = os.path.join(directory, data_folder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# =========================
# Parse log regex + mode markers
# =========================
accel_pat = re.compile(r"&?X\s*([-+]?\d*\.?\d+)\s*Y\s*([-+]?\d*\.?\d+)\s*Z\s*([-+]?\d*\.?\d+)")
gyro_pat  = re.compile(r"&?GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)")
mode_start = re.compile(r"MODE\s*=\s*GYRO_START", re.I)
mode_end   = re.compile(r"MODE\s*=\s*GYRO_END", re.I)

def parse_log_with_indices(path: str):
    accel_samples = []
    gyro_segments = []
    bias = None
    global_idx = 0
    in_seg = False
    seg_buf = []
    seg_start_idx = None

    gyro_bias_pat = re.compile(
        r"&?GYRO_BIAS.*?GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)",
        re.I
    )

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_bias = gyro_bias_pat.search(line)
            if m_bias:
                bias = np.array(list(map(float, m_bias.groups())))
                continue

            if mode_start.search(line):
                in_seg = True
                seg_buf = []
                seg_start_idx = global_idx
                continue

            if mode_end.search(line):
                in_seg = False
                seg_end_idx = global_idx
                gyro_segments.append({
                    "start_idx": seg_start_idx,
                    "end_idx": seg_end_idx,
                    "samples": np.array(seg_buf, dtype=float)
                })
                seg_buf = []
                seg_start_idx = None
                continue

            m_g = gyro_pat.search(line)
            if m_g:
                vals = list(map(float, m_g.groups()))
                if in_seg:
                    seg_buf.append(vals)
                global_idx += 1
                continue

            m_a = accel_pat.search(line)
            if m_a:
                vals = np.array(list(map(float, m_a.groups())), dtype=float)
                accel_samples.append((global_idx, vals))
                global_idx += 1
                continue

    if in_seg and seg_buf:
        seg_end_idx = global_idx
        gyro_segments.append({
            "start_idx": seg_start_idx,
            "end_idx": seg_end_idx,
            "samples": np.array(seg_buf, dtype=float)
        })

    return accel_samples, gyro_segments, bias

def accel_samples_to_indexed_array(accel_samples):
    if not accel_samples:
        return np.array([], dtype=int), np.zeros((0,3), dtype=float)
    idxs = np.array([i for i, v in accel_samples], dtype=int)
    vals = np.vstack([v for i, v in accel_samples]).astype(float)
    return idxs, vals

def average_accel_window(idxs, vals, center_idx, before=10, after=0, prefer_side=None):
    """
    Return mean accel vector using samples in window:
      [center_idx - before, center_idx + after)
    idxs: array of sample indices
    vals: array of accel values (same length)
    prefer_side: None (default) -> fallback to nearest sample (old behavior)
                 'before'          -> fallback to last sample with idx < center_idx
                 'after'           -> fallback to first sample with idx >= center_idx

    This makes behaviour robust when accel is sparse (MCU averaged per-static).
    """
    low = center_idx - before
    high = center_idx + after
    mask = (idxs >= low) & (idxs < high)
    if np.any(mask):
        return np.mean(vals[mask], axis=0)

    # no samples inside the desired window -> fallbacks
    if prefer_side == 'before':
        # choose the last accel sample strictly before center_idx (if any)
        candidates = np.where(idxs < center_idx)[0]
        if candidates.size:
            idx = candidates[-1]
            return vals[idx]
    elif prefer_side == 'after':
        # choose the first accel sample at/after center_idx (if any)
        candidates = np.where(idxs >= center_idx)[0]
        if candidates.size:
            idx = candidates[0]
            return vals[idx]

    # old fallback: nearest sample (keeps backwards compatibility)
    nearest = np.argmin(np.abs(idxs - center_idx))
    return vals[nearest]

# =========================
# Accel calibration (unchanged)
# =========================
def build_Ta(a_yz,a_zy,a_zx): return np.array([[1,-a_yz,a_zy],[0,1,-a_zx],[0,0,1]])
def build_K(sx,sy,sz): return np.diag([sx,sy,sz])

def accel_resid(p,data):
    bx,by,bz,sx,sy,sz,a_yz,a_zy,a_zx=p
    b=np.array([bx,by,bz]); K=build_K(sx,sy,sz); T=build_Ta(a_yz,a_zy,a_zx)
    corr=(T@(K@(data.T-b[:,None]))).T
    return np.linalg.norm(corr,axis=1)-g

def accel_find_params(data):
    init=np.array([0,0,0,1,1,1,0,0,0],float)
    return least_squares(accel_resid,init,args=(data,)).x

def accel_apply(data,p):
    bx,by,bz,sx,sy,sz,a_yz,a_zy,a_zx=p
    b=np.array([bx,by,bz]); K=build_K(sx,sy,sz); T=build_Ta(a_yz,a_zy,a_zx)
    return (T@(K@(data.T-b[:,None]))).T

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
# Projection & normalization
# =========================
def project_onto_plane(v, n):
    return v - np.dot(v, n) * n

def safe_normalize(v, eps=1e-12):
    norm = np.linalg.norm(v)
    if norm <= eps:
        return v, norm
    return v / norm, norm

# =========================
# Gyro helpers
# =========================
def build_Tg(g_yz,g_zy,g_xz,g_zx,g_xy,g_yx):
    return np.array([[1,-g_yz,g_zy],[g_xz,1,-g_zx],[-g_xy,g_yx,1]])

def omega_to_rad_s(omega):
    return np.deg2rad(omega) if GYRO_UNITS.lower()=="deg_s" else omega

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def rk4n_integrate_quat(omegas_rad_s, dt):
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for w in omegas_rad_s:
        def f(qv):
            omega_q = np.array([0.0, *w])
            return 0.5 * quat_multiply(qv, omega_q)
        k1 = f(q)
        k2 = f(q + 0.5*dt*k1)
        k3 = f(q + 0.5*dt*k2)
        k4 = f(q + dt*k3)
        q = q + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        q = quat_normalize(q)
    return q

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])

def skew(v):
    x,y,z = v
    return np.array([[0.0, -z,   y],
                     [z,   0.0, -x],
                     [-y,  x,  0.0]])

def Tg_from_eps(eps):
    theta = np.linalg.norm(eps)
    if theta < 1e-12:
        return np.eye(3) + skew(eps)
    k = eps / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def rotvec_to_R(rv):
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

# =========================
# Gyro residuals & calibrator (improved)
# =========================
def build_gyro_residuals(accel_dirs, gyro_segments, params):
    idx = 0
    if GYRO_FIT_MISALIGNMENT:
        g_yz, g_zy, g_xz, g_zx, g_xy, g_yx = params[idx:idx+6]; idx += 6
        Tg = build_Tg(g_yz, g_zy, g_xz, g_zx, g_xy, g_yx)
    else:
        Tg = np.eye(3)

    # scales (may be in params or identity if not provided)
    sgx, sgy, sgz = params[idx:idx+3]; idx += 3
    Kg = np.diag([sgx, sgy, sgz])

    # bias (either from params or zero)
    bg = np.array(params[idx:idx+3]) if idx < len(params) else np.zeros(3)

    dt = 1.0 / SAMPLE_RATE_HZ
    res = []

    for k in range(1, len(accel_dirs)):
        a0, a1 = accel_dirs[k-1], accel_dirs[k]
        seg = gyro_segments[k-1]
        if seg.size == 0:
            continue

        seg_arr = np.array(seg, dtype=float)
        seg_demean = seg_arr - bg[None, :]
        seg_scaled = (Kg @ seg_demean.T).T
        omega_body = (Tg @ seg_scaled.T).T
        omega_r = omega_to_rad_s(omega_body)

        if GYRO_INTEGRATOR == "rk4":
            q_final = rk4n_integrate_quat(omega_r, dt)
            R = quat_to_R(q_final)
            a_pred = R @ a0
        elif GYRO_INTEGRATOR == "rodrigues":
            dtheta = np.sum(omega_r, axis=0) * dt
            theta = np.linalg.norm(dtheta)
            if theta < 1e-12:
                R = np.eye(3)
            else:
                k_axis = dtheta / theta
                K = np.array([[0, -k_axis[2], k_axis[1]],
                              [k_axis[2], 0, -k_axis[0]],
                              [-k_axis[1], k_axis[0], 0]])
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            a_pred = R @ a0
        else:
            dtheta = np.sum(omega_r, axis=0) * dt
            a_pred = a0 - np.cross(dtheta, a0)

        res.append(a_pred - a1)

    return np.concatenate(res) if res else np.zeros(0)

def calibrate_gyro(accel_calib, gyro_segments, static_bias=None, bias_guess=None, init_scale=None):
    dirs = accel_calib / np.linalg.norm(accel_calib, axis=1, keepdims=True)
    p = []; lb = []; ub = []

    if GYRO_FIT_MISALIGNMENT:
        p += [0.0]*6
        lb += [-0.5]*6
        ub += [ 0.5]*6

    # initialize scales
    if init_scale is None:
        p += [1.0, 1.0, 1.0]
    else:
        p += [init_scale, init_scale, init_scale]
    lb += [0.5, 0.5, 0.5]
    ub += [2.0, 2.0, 2.0]

    use_static_bias = static_bias is not None
    if not use_static_bias and GYRO_CALIB_MODE == "fit_bias":
        # bias bounds depend on units
        if GYRO_UNITS.lower() == "deg_s":
            bias_bound = 200.0
        else:
            bias_bound = 10.0
        if bias_guess is None:
            p += [0.0, 0.0, 0.0]
        else:
            p += list(bias_guess)
        lb += [-bias_bound, -bias_bound, -bias_bound]
        ub += [ bias_bound,  bias_bound,  bias_bound]

    p = np.array(p, float); lb = np.array(lb, float); ub = np.array(ub, float)

    def fun(x):
        x_full = np.concatenate([x, static_bias]) if use_static_bias else x
        return build_gyro_residuals(dirs, gyro_segments, x_full)

    print(f"Starting gyro optimization (rotvec misalign={GYRO_FIT_MISALIGNMENT})...")
    res = least_squares(fun, p, bounds=(lb, ub), method='trf', max_nfev=2000,
                        verbose=2, loss='soft_l1', ftol=1e-6, xtol=1e-6, gtol=1e-6, x_scale='jac')

    out = {"success": res.success, "cost": 0.5 * np.sum(res.fun ** 2)}
    i = 0
    if GYRO_FIT_MISALIGNMENT:
        out["gamma"] = res.x[i:i+6]; i += 6
    out["scale"] = res.x[i:i+3]; i += 3
    if use_static_bias:
        out["bias"] = static_bias
    elif GYRO_CALIB_MODE == "fit_bias":
        out["bias"] = res.x[i:i+3]
    else:
        out["bias"] = np.zeros(3)
    return res.x, out

# =========================
# Visualizer (unchanged)
# =========================
def plot_accel_sphere(raw_ms2: np.ndarray, calib_ms2: np.ndarray):
    fig = plt.figure(figsize=(10, 5))
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = g * np.outer(np.cos(u), np.sin(v))
    ys = g * np.outer(np.sin(u), np.sin(v))
    zs = g * np.outer(np.ones_like(u), np.cos(v))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(raw_ms2[:, 0], raw_ms2[:, 1], raw_ms2[:, 2],
                s=6, alpha=0.6, color="tab:red", label="Raw data")
    ax1.plot_wireframe(xs, ys, zs, color="gray", alpha=0.25, label="Reference sphere")
    ax1.set_title("Raw accelerometer data")
    ax1.set_xlabel("X [m/s²]"); ax1.set_ylabel("Y [m/s²]"); ax1.set_zlabel("Z [m/s²]")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(calib_ms2[:, 0], calib_ms2[:, 1], calib_ms2[:, 2],
                s=6, alpha=0.6, color="tab:blue", label="Calibrated data")
    ax2.plot_wireframe(xs, ys, zs, color="gray", alpha=0.25, label="Reference sphere")
    ax2.set_title("Calibrated accelerometer data")
    ax2.set_xlabel("X [m/s²]"); ax2.set_ylabel("Y [m/s²]"); ax2.set_zlabel("Z [m/s²]")
    ax2.legend()
    plt.tight_layout(); plt.show()

# =========================
# Pose computations (unchanged)
# =========================
def compute_segment_pose_vectors(accel_samples, gyro_segments, accel_params, win_before=10, win_after=10):
    idxs, vals = accel_samples_to_indexed_array(accel_samples)
    if idxs.size == 0:
        return []

    mean_norm = np.mean(np.linalg.norm(vals, axis=1))
    vals_ms2 = vals * g if 0.3 <= mean_norm <= 2 else vals
    vals_cal = accel_apply(vals_ms2, accel_params)

    poses = []
    for seg in gyro_segments:
        sidx = int(seg["start_idx"])
        eidx = int(seg["end_idx"])
        # prefer accel sample before the start for a0, and after the end for a1
        a0_raw = average_accel_window(idxs, vals_cal, sidx, before=win_before, after=0, prefer_side='before')
        a1_raw = average_accel_window(idxs, vals_cal, eidx, before=0, after=win_after, prefer_side='after')
        na0 = a0_raw / np.linalg.norm(a0_raw) if np.linalg.norm(a0_raw) > 0 else a0_raw
        na1 = a1_raw / np.linalg.norm(a1_raw) if np.linalg.norm(a1_raw) > 0 else a1_raw
        poses.append((na0, na1))
    return poses

def integrate_segment_to_rotation_for_compare(seg, sample_rate, integrator="rk4", bias=None):
    seg_arr = np.array(seg, dtype=float)
    if seg_arr.size == 0:
        return np.array([1.0,0.0,0.0,0.0]), np.eye(3), 0.0
    if bias is not None:
        seg_arr = seg_arr - np.array(bias)[None, :]
    omega_r = omega_to_rad_s(seg_arr)
    dt = 1.0 / sample_rate
    if integrator == "rk4":
        q = rk4n_integrate_quat(omega_r, dt)
        R = quat_to_R(q)
        w = np.clip(q[0], -1.0, 1.0)
        ang = 2.0 * np.arccos(w)
        return q, R, ang
    elif integrator == "rodrigues":
        dtheta = np.sum(omega_r, axis=0) * dt
        theta = np.linalg.norm(dtheta)
        if theta < 1e-12:
            return np.array([1.0,0,0,0]), np.eye(3), 0.0
        k_axis = dtheta / theta
        K = np.array([[0, -k_axis[2], k_axis[1]],
                      [k_axis[2], 0, -k_axis[0]],
                      [-k_axis[1], k_axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return None, R, theta
    else:
        dtheta = np.sum(omega_r, axis=0) * dt
        theta = np.linalg.norm(dtheta)
        K = np.array([[0, -dtheta[2], dtheta[1]],
                      [dtheta[2], 0, -dtheta[0]],
                      [-dtheta[1], dtheta[0], 0]])
        R = np.eye(3) + K
        return None, R, theta

def compare_tedaldi_style(accel_samples, gyro_segments, accel_params, sample_rate, integrator="rk4", bias=None, win_before=10, win_after=10, limit=200):
    poses = compute_segment_pose_vectors(accel_samples, gyro_segments, accel_params, win_before=win_before, win_after=win_after)
    results = []
    n = min(len(poses), len(gyro_segments), limit)
    for i in range(n):
        a0, a1 = poses[i]
        seg = gyro_segments[i]["samples"]
        q, R, gyro_ang = integrate_segment_to_rotation_for_compare(seg, sample_rate, integrator=integrator, bias=bias)

        g_hat, g_norm = safe_normalize(a0)
        if g_norm < 1e-6:
            print(f"Seg {i}: skipping (invalid gravity vector)")
            continue

        a0_plane = project_onto_plane(a0, g_hat)
        a1_plane = project_onto_plane(a1, g_hat)
        a0p, n0 = safe_normalize(a0_plane)
        a1p, n1 = safe_normalize(a1_plane)
        if n0 < 1e-6 or n1 < 1e-6:
            print(f"Seg {i}: skipping (projection too small; a0 or a1 approx parallel to gravity)")
            continue

        dot_accel = np.clip(np.dot(a0p, a1p), -1.0, 1.0)
        accel_ang_plane = np.arccos(dot_accel)

        a1_from_gyro = R @ a0
        a1g_plane = project_onto_plane(a1_from_gyro, g_hat)
        a1g_p, n1g = safe_normalize(a1g_plane)
        if n1g < 1e-6:
            print(f"Seg {i}: skipping (gyro predicted projection too small)")
            continue

        dot_pred = np.clip(np.dot(a1g_p, a1p), -1.0, 1.0)
        pred_angle_plane = np.arccos(dot_pred)
        gyro_plane_angle = np.arccos(np.clip(np.dot(a0p, a1g_p), -1.0, 1.0))

        results.append({
            "idx": i,
            "accel_angle_rad": accel_ang_plane,
            "gyro_angle_rad": gyro_plane_angle,
            "pred_error_rad": pred_angle_plane,
            "a0": a0,
            "a1": a1,
            "a1_pred": a1_from_gyro
        })
        print(f"Seg {i:2d}: accel_plane_angle={np.degrees(accel_ang_plane):7.3f}° | gyro_plane_angle={np.degrees(gyro_plane_angle):7.3f}° | pred_error={np.degrees(pred_angle_plane):7.3f}°")
    return results

def compare_cross_calibration(target_path: str, calib_params: np.ndarray):
    print("\n=== Cross-dataset calibration test ===")
    print(f"Using parameters from reference run on: {os.path.basename(target_path)}")
    accel_samples, _, _ = parse_log_with_indices(target_path)
    idxs, vals = accel_samples_to_indexed_array(accel_samples)
    if vals.size == 0:
        print("No accelerometer data found in target file.")
        return
    mean_norm = np.mean(np.linalg.norm(vals, axis=1))
    accel_new_ms2 = vals * g if 0.3 <= mean_norm <= 2 else vals
    accel_calib_ms2 = accel_apply(accel_new_ms2, calib_params)
    accel_summary(accel_calib_ms2, accel_new_ms2)

# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--accel-file", type=str, default=None)
    ap.add_argument("--accel-params", type=str, default=None)
    ap.add_argument("--integrator", type=str, choices=["rk4","rodrigues","smallangle"], default="rk4")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--units", type=str, choices=["deg_s","rad_s"], default=None)
    args = ap.parse_args()

    if args.rate is not None:
        globals()['SAMPLE_RATE_HZ'] = args.rate
    if args.units is not None:
        globals()['GYRO_UNITS'] = args.units
    if args.integrator:
        globals()['GYRO_INTEGRATOR'] = args.integrator

    if args.file is not None:
        path = args.file
    else:
        txts = list_txt_files(".")
        if not txts:
            sys.exit("No .txt files found.")
        if len(txts) == 1:
            path = os.path.join(data_folder, txts[0])
        else:
            for i, f in enumerate(txts):
                print(f"{i}: {f}")
            path = os.path.join(data_folder, txts[int(input('Select: '))])

    accel_samples, gyro_segments, bias_from_log = parse_log_with_indices(path)

    # --- enforce axis sign on bias and raw samples (important) ---
    if bias_from_log is not None:
        bias_from_log = np.array(bias_from_log, dtype=float) * GYRO_AXIS_SIGN
    for seg in gyro_segments:
        if getattr(seg, "samples", None) is not None and np.array(seg["samples"]).size:
            seg["samples"] = (np.array(seg["samples"], dtype=float) * GYRO_AXIS_SIGN).tolist()

    print("=== GYRO PARSING DIAGNOSTICS ===")
    print("GYRO_UNITS =", GYRO_UNITS)
    print("Parsed bias_from_log (raw):", bias_from_log, type(bias_from_log))
    if gyro_segments and len(gyro_segments)>0 and np.array(gyro_segments[0]["samples"]).size:
        s0 = np.array(gyro_segments[0]["samples"], dtype=float)
        print("first raw gyro sample (sensor units):", s0[0])
        if bias_from_log is not None:
            print("first sample - bias (sensor units):", s0[0] - bias_from_log)
            print("after conversion to rad/s:", omega_to_rad_s(s0[0] - bias_from_log))
        else:
            print("after conversion to rad/s (no bias):", omega_to_rad_s(s0[0]))
    print("Apply axis sign variable (GYRO_AXIS_SIGN):", GYRO_AXIS_SIGN)
    print("=== END DIAGNOSTICS ===")

    if args.accel_file:
        accel_samples_accel_file, _, _ = parse_log_with_indices(args.accel_file)
        if accel_samples_accel_file:
            accel_samples = accel_samples_accel_file

    acc_p = None
    if args.accel_params:
        try:
            if os.path.isfile(args.accel_params):
                if args.accel_params.endswith(".npy"):
                    acc_p = np.load(args.accel_params)
                else:
                    acc_p = np.loadtxt(args.accel_params)
                print(f"\nLoaded accelerometer parameters from file {args.accel_params}:")
            else:
                values = [float(v.strip()) for v in args.accel_params.split(",")]
                if len(values) != 9:
                    raise ValueError("Expected 9 parameters (bx,by,bz,sx,sy,sz,a_yz,a_zy,a_zx)")
                acc_p = np.array(values, dtype=float)
                print("\nLoaded accelerometer parameters from command line:")
            print(" One row:", " ".join(f"{x:.8e}" for x in acc_p))
        except Exception as e:
            print(f"Failed to parse --accel-params: {e}")
            sys.exit(1)

    idxs, accel_vals = accel_samples_to_indexed_array(accel_samples)
    if accel_vals.size == 0:
        sys.exit("No accel data found in the parsed logs.")

    mean_norm = np.mean(np.linalg.norm(accel_vals, axis=1))
    accel_raw_all = accel_vals * g if 0.3 <= mean_norm <= 2 else accel_vals

    if acc_p is None:
        acc_p = accel_find_params(accel_raw_all)
        print("\nComputed accelerometer calibration:")
        print(" Bias:", acc_p[:3])
        print(" Scales:", acc_p[3:6])
        print(" Misalignment (a_yz, a_zy, a_zx):", acc_p[6:])
        print(" One row:", " ".join(f"{x:.8e}" for x in acc_p))
        accel_cal_all = accel_apply(accel_raw_all, acc_p)
        accel_summary(accel_cal_all, accel_raw_all)
    else:
        print("\nUsing provided accelerometer calibration parameters.")
        accel_cal_all = accel_apply(accel_raw_all, acc_p)

    compare_cross_calibration("DATA/big_turntable_test_sample.txt", acc_p)

    print("\nRunning Tedaldi-style comparisons (start/end accel vectors vs integrated gyro angles, ignoring rotation about gravity)...")
    compare_results = compare_tedaldi_style(
        accel_samples=accel_samples,
        gyro_segments=gyro_segments,
        accel_params=acc_p,
        sample_rate=SAMPLE_RATE_HZ,
        integrator=GYRO_INTEGRATOR,
        bias=bias_from_log,
        win_before=10,
        win_after=10,
        limit=500
    )

    poses = compute_segment_pose_vectors(accel_samples, gyro_segments, acc_p, win_before=10, win_after=10)
    if poses:
        accel_cal_list = []
        accel_cal_list.append(poses[0][0])
        for (a0, a1) in poses:
            accel_cal_list.append(a1)
        accel_cal = np.vstack(accel_cal_list)
    else:
        accel_cal = accel_cal_all

    gyros = [seg["samples"] for seg in gyro_segments]

    if gyros:
        print(f"\nFound {len(gyros)} gyro segments.")
        if len(gyros) != len(accel_cal) - 1:
            n = min(len(gyros), len(accel_cal) - 1)
            gyros = gyros[:n]
            accel_cal = accel_cal[:n + 1]

        print("Filtering gyro segments for real motion...")
        static_bias = bias_from_log if bias_from_log is not None and getattr(bias_from_log, "size", 0) == 3 else None
        if static_bias is not None:
            bias_est = static_bias
        else:
            all_samples = np.vstack([seg for seg in gyros if np.array(seg).size > 0]) if gyros else np.zeros((0,3))
            bias_est = np.mean(all_samples, axis=0) if all_samples.size else np.zeros(3)
        print(f"Estimated gyro bias for filtering/seed: {bias_est}")

        # Diagnostics: compute median gyro vs accel integrated angles to seed scale
        gyro_angles = []
        for seg in gyros[:min(len(gyros),200)]:
            if np.array(seg).size == 0: continue
            _, _, ang_g = integrate_segment_to_rotation_for_compare(seg, SAMPLE_RATE_HZ,
                                                                    integrator=GYRO_INTEGRATOR,
                                                                    bias=bias_est)
            gyro_angles.append(ang_g)
        accel_angles = [r["accel_angle_rad"] for r in compare_results[:len(gyro_angles)] if r["accel_angle_rad"]>1e-9]
        scale_guess = 1.0
        if gyro_angles and accel_angles:
            med_g = float(np.median(gyro_angles))
            med_a = float(np.median(accel_angles))
            if med_g > 1e-9:
                scale_guess = float(med_a / (med_g + 1e-12))
            # clamp to safe range
            scale_guess = float(np.clip(scale_guess, 0.5, 2.0))
        print("scale_guess (seed):", scale_guess)

        # Print sample durations to sanity-check sample rate
        for i, seg in enumerate(gyros[:8]):
            n = np.array(seg).shape[0]
            print(f"seg {i}: samples={n}, duration={n / SAMPLE_RATE_HZ:.3f}s")

        # Call calibrator with seeding
        params_vec, gyro_info = calibrate_gyro(accel_cal, gyros,
                                               static_bias=static_bias,
                                               bias_guess=(bias_est if static_bias is None else None),
                                               init_scale=scale_guess)

        vals = list(params_vec)
        bias = np.array(gyro_info["bias"], dtype=float)
        vals += bias.tolist()

        labels = []
        if GYRO_FIT_MISALIGNMENT:
            labels += ["gamma_yz", "gamma_zy", "gamma_xz", "gamma_zx", "gamma_xy", "gamma_yx"]
        labels += ["sgx", "sgy", "sgz"]
        labels += ["bgx", "bgy", "bgz"]

        print("One row params:",
            " ".join(f"{lab}={v:.8e}" for lab, v in zip(labels, vals)))
        print("One row (values):",
            " ".join(f"{v:.8e}" for v in vals))