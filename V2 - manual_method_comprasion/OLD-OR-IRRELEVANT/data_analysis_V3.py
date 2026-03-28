import os, re, sys, argparse
import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D plotting)

# =========================
# Configuration
# =========================
data_folder = "DATA"
g = 9.81872
SAMPLE_RATE_HZ = 70.0
GYRO_UNITS = "deg_s"
GYRO_AXIS_SIGN = np.array([1.0, 1.0, 1.0])
GYRO_CALIB_MODE = "fit_bias"
GYRO_FIT_MISALIGNMENT = False
ACCEL_UNITS = "g"

# Integrator options: 'rk4' (default), 'rodrigues', 'smallangle'
GYRO_INTEGRATOR = "rk4"

# =========================
# Helpers
# =========================

def angle_between_vectors_rad(v1, v2, eps=1e-12):
    """
    Returns the angle between v1 and v2 in radians.
    Robust against tiny floating point errors.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < eps or n2 < eps:
        raise ValueError("Zero-length vector passed to angle_between_vectors_rad")

    u1 = v1 / n1
    u2 = v2 / n2

    c = np.dot(u1, u2)
    c = np.clip(c, -1.0, 1.0)

    return np.arccos(c)


def angle_between_vectors_deg(v1, v2, eps=1e-12):
    """
    Returns the angle between v1 and v2 in degrees.
    """
    return np.degrees(angle_between_vectors_rad(v1, v2, eps=eps))


def list_txt_files(directory):
    folder_path = os.path.join(directory, data_folder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# -------------------------
# small math helpers used in several places
# -------------------------
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y,  x,  0.0]])

def rodrigues_exp(phi):
    """
    Return rotation matrix exp(skew(phi)). phi is a 3-vector in radians.
    Small-theta series used for numerical stability.
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + skew(phi)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def per_segment_errors(scales, bias, dirs, gyros, gamma=np.zeros(6)):
    """
    scales: (3,) sgx,sgy,sgz
    bias: (3,) fixed bias
    dirs: list/array of accel unit vectors
    gyros: list of segments (Nx3 arrays)
    gamma: (6,) misalignment vector [g_yz, g_zy, g_xz, g_zx, g_xy, g_yx]
    """
    sgx, sgy, sgz = scales
    Kg = np.diag([sgx, sgy, sgz])
    Tg = build_Tg(*gamma)

    seg_errs = []
    for k in range(1, len(dirs)):
        a0, a1 = dirs[k-1], dirs[k]
        seg = gyros[k-1]
        if seg.size == 0:
            seg_errs.append(np.nan)
            continue

        omega = (Tg @ (Kg @ (seg.T + bias[:, None]))).T
        omega_r = omega_to_rad_s(omega)

        dt = 1.0 / SAMPLE_RATE_HZ

        theta = np.linalg.norm(np.sum(omega_r, axis=0) * dt)
        print("segment rotation:", np.degrees(theta), "deg")    

        q_final = rk4n_integrate_quat(omega_r, 1.0 / SAMPLE_RATE_HZ)
        R = quat_to_R(q_final)

        err_vec = (R @ a0) - a1
        seg_errs.append(np.linalg.norm(err_vec))

    return np.array(seg_errs)

# =========================
# Parse log regex + mode markers
# =========================
accel_pat = re.compile(r"&?X\s*([-+]?\d*\.?\d+)\s*Y\s*([-+]?\d*\.?\d+)\s*Z\s*([-+]?\d*\.?\d+)")
gyro_pat  = re.compile(r"&?GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)")
mode_start = re.compile(r"MODE\s*=\s*GYRO_START", re.I)
mode_end   = re.compile(r"MODE\s*=\s*GYRO_END", re.I)

# -------------------------
# parse_log_with_indices + helpers for aligned parsing
# -------------------------
def parse_log_with_indices(path: str):
    accel_samples = []
    gyro_segments = []
    bias = None

    global_idx = 0
    in_seg = False
    pending_end = False
    seg_buf = []

    seg_start_accel_idx = None
    last_accel_idx = None

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
                bias = np.array(list(map(float, m_bias.groups())), dtype=float)
                continue

            if mode_start.search(line):
                in_seg = True
                pending_end = False
                seg_buf = []
                seg_start_accel_idx = last_accel_idx
                continue

            if mode_end.search(line):
                in_seg = False
                pending_end = True
                continue

            m_g = gyro_pat.search(line)
            if m_g:
                vals = np.array(list(map(float, m_g.groups())), dtype=float)
                if in_seg:
                    seg_buf.append(vals)
                global_idx += 1
                continue

            m_a = accel_pat.search(line)
            if m_a:
                vals = np.array(list(map(float, m_a.groups())), dtype=float)
                accel_samples.append((global_idx, vals))
                last_accel_idx = global_idx

                if pending_end and seg_start_accel_idx is not None:
                    gyro_segments.append({
                        "start_accel_idx": seg_start_accel_idx,
                        "end_accel_idx": global_idx,
                        "samples": np.array(seg_buf, dtype=float)
                    })
                    seg_buf = []
                    seg_start_accel_idx = None
                    pending_end = False

                global_idx += 1
                continue

    return accel_samples, gyro_segments, bias

def accel_samples_to_indexed_array(accel_samples):
    """
    Convert accel_samples list [(idx, vec), ...] to
    two arrays: indices (N,) and values (N,3)
    """
    if not accel_samples:
        return np.array([], dtype=int), np.zeros((0,3), dtype=float)
    idxs = np.array([i for i, v in accel_samples], dtype=int)
    vals = np.vstack([v for i, v in accel_samples]).astype(float)
    return idxs, vals

def average_accel_window(idxs, vals, center_idx, before=10, after=0):
    """
    Return mean accel vector using samples in window:
      [center_idx - before, center_idx + after)
    idxs: array of sample indices
    vals: array of accel values (same length)
    """
    low = center_idx - before
    high = center_idx + after
    mask = (idxs >= low) & (idxs < high)
    if not np.any(mask):
        # fallback: choose nearest sample
        nearest = np.argmin(np.abs(idxs - center_idx))
        return vals[nearest]
    return np.mean(vals[mask], axis=0)

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
# Projection helpers (ignore gravity yaw)
# =========================
def project_onto_plane(v, n):
    """Project vector v onto plane orthogonal to unit vector n."""
    return v - np.dot(v, n) * n

def safe_normalize(v, eps=1e-3):
    norm = np.linalg.norm(v)
    if norm <= eps:
        return v, norm
    return v / norm, norm

# =========================
# Gyro helpers (integrators) & quaternion utilities
# =========================
def build_Tg(g_yz,g_zy,g_xz,g_zx,g_xy,g_yx):
    return np.array([[1,-g_yz,g_zy],[g_xz,1,-g_zx],[-g_xy,g_yx,1]])

def omega_to_rad_s(omega):
    return np.deg2rad(omega) if GYRO_UNITS.lower()=="deg_s" else omega

def quat_omega_matrix(omega):
    wx, wy, wz = omega
    return np.array([
        [0.0,   -wx,   -wy,   -wz],
        [wx,    0.0,    wz,   -wy],
        [wy,   -wz,    0.0,    wx],
        [wz,    wy,   -wx,    0.0],
    ])

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

# =========================
# Gyro calibration (residuals) unchanged conceptually
# =========================
# helper: skew and Rodrigues-exp for small-angle -> rotation matrix

# modified build_gyro_residuals that accepts phi (3) or full gamma(6) depending on flag
def build_gyro_residuals_with_reg(accel_dirs, gyro_segments, params, fit_misalignment=True, reg_sigma=None, bias_prior=None):
    """
    params layout when fit_misalignment:
      [phi_x,phi_y,phi_z, sgx,sgy,sgz, bgx,bgy,bgz?]
    when not fit_misalignment:
      [sgx, sgy, sgz, bgx,bgy,bgz?]
    reg_sigma: dict with keys 'phi','scale','bias' providing sigma values for regularization
    bias_prior: (3,) prior estimate to penalize bias away from (optional)
    """
    idx = 0
    if fit_misalignment:
        phi = np.array(params[idx:idx+3]); idx += 3
        Tg = rodrigues_exp(phi)
    else:
        Tg = np.eye(3)

    sgx, sgy, sgz = params[idx:idx+3]; idx += 3
    Kg = np.diag([sgx, sgy, sgz])
    # bias may appear in params if provided:
    bg = np.array(params[idx:idx+3]) if idx + 3 <= len(params) else np.zeros(3)
    dt = 1.0 / SAMPLE_RATE_HZ

    motion_res = []
    for k in range(1, len(accel_dirs)):
        a0, a1 = accel_dirs[k-1], accel_dirs[k]
        seg = gyro_segments[k-1]
        if seg.size == 0:
            continue
        omega = (Tg @ (Kg @ (seg.T + bg[:, None]))).T
        omega_r = omega_to_rad_s(omega)

        # integrate with your chosen integrator (reuse existing code patterns)
        if GYRO_INTEGRATOR == "rk4":
            q_final = rk4n_integrate_quat(omega_r, dt)
            R = quat_to_R(q_final)
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
        else:  # smallangle
            dtheta = np.sum(omega_r, axis=0) * dt
            K = np.array([[0, -dtheta[2], dtheta[1]],
                          [dtheta[2], 0, -dtheta[0]],
                          [-dtheta[1], dtheta[0], 0]])
            R = np.eye(3) + K

        a_pred = (R @ a0) - a1
        motion_res.append(a_pred)

    motion_res = np.concatenate(motion_res) if motion_res else np.zeros(0)

    # build regularizer residuals (small)
    reg = []
    if reg_sigma is None:
        reg_sigma = {'phi':1e-2, 'scale':1e-2, 'bias':1e-2}

    if fit_misalignment:
        # penalize phi away from zero: residual = phi / sigma_phi
        reg.append(phi / reg_sigma['phi'])


    # penalize bias away from bias_prior if provided, otherwise small bias:
    if bias_prior is not None:
        reg.append((bg - bias_prior) / reg_sigma['bias'])
    else:
        reg.append(bg / reg_sigma['bias'])

    reg = np.concatenate(reg) if reg else np.zeros(0)
    return np.concatenate([motion_res, reg])

# small wrapper used by least_squares
def fun_for_lsq(x, accel_dirs, gyro_segments, fit_misalignment=True, reg_sigma=None, bias_prior=None):
    return build_gyro_residuals_with_reg(accel_dirs, gyro_segments, x, fit_misalignment=fit_misalignment, reg_sigma=reg_sigma, bias_prior=bias_prior)

def calibrate_gyro(accel_calib, gyro_segments, static_bias=None):
    """
    Full 6-parameter misalignment fit (no small-angle phi).
    Bias is fixed.
    Params layout (stage2):
        [g_yz, g_zy, g_xz, g_zx, g_xy, g_yx,  sgx, sgy, sgz]
    """
    dirs = accel_calib / np.linalg.norm(accel_calib, axis=1, keepdims=True)

    # ---- FIXED bias ----
    if static_bias is not None and getattr(static_bias, "size", 0) == 3:
        bias_fixed = np.array(static_bias, dtype=float)
    else:
        all_samples = np.vstack([seg for seg in gyro_segments if seg.size > 0]) if gyro_segments else np.zeros((0,3))
        bias_fixed = np.mean(all_samples, axis=0) if all_samples.size else np.zeros(3)

    dt = 1.0 / SAMPLE_RATE_HZ

    # Core motion residual (6 misalign + 3 scales)
    def motion_residual(accel_dirs, gyro_segments, params):
        idx = 0
        g_yz, g_zy, g_xz, g_zx, g_xy, g_yx = params[idx:idx+6]; idx += 6
        Tg = build_Tg(g_yz, g_zy, g_xz, g_zx, g_xy, g_yx)

        sgx, sgy, sgz = params[idx:idx+3]
        Kg = np.diag([sgx, sgy, sgz])

        bg = bias_fixed

        res_list = []
        for k in range(1, len(accel_dirs)):
            a0, a1 = accel_dirs[k-1], accel_dirs[k]
            seg = gyro_segments[k-1]
            if seg.size == 0:
                continue

            omega = (Tg @ (Kg @ (seg.T + bg[:, None]))).T
            omega_r = omega_to_rad_s(omega)

            if GYRO_INTEGRATOR == "rk4":
                q_final = rk4n_integrate_quat(omega_r, dt)
                R = quat_to_R(q_final)
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
            else:
                dtheta = np.sum(omega_r, axis=0) * dt
                K = np.array([[0, -dtheta[2], dtheta[1]],
                              [dtheta[2], 0, -dtheta[0]],
                              [-dtheta[1], dtheta[0], 0]])
                R = np.eye(3) + K

            res_list.append((R @ a0) - a1)

        return np.concatenate(res_list) if res_list else np.zeros(0)

    # Stage 1 — fit scales only (misalign=0)
    def resid_stage1(scales):
        params = np.concatenate([np.zeros(6), scales])
        return motion_residual(dirs, gyro_segments, params)

    x0_stage1 = np.ones(3)
    lb1 = np.full(3, 0.5)
    ub1 = np.full(3, 1.5)

    print("Stage 1: fitting scales (misalign fixed=0) ...")
    res1 = least_squares(resid_stage1, x0_stage1,
                         bounds=(lb1, ub1),
                         verbose=2,
                         loss='soft_l1',
                         x_scale='jac')

    scales1 = res1.x.copy()
    cost1 = 0.5 * np.sum(res1.fun**2)
    print(" Stage1 cost:", cost1, " scales:", scales1)

    # Stage 2 — fit 6 misalign + 3 scales
    def resid_stage2(params):
        motion = motion_residual(dirs, gyro_segments, params)

        # weak regularization on misalignment
        gamma = params[:6]
        sigma_gamma = 0.02
        reg = gamma / sigma_gamma

        return np.concatenate([motion, reg])

    x0_stage2 = np.concatenate([np.zeros(6), scales1])

    lb2 = np.concatenate([np.full(6, -0.2), np.full(3, 0.5)])
    ub2 = np.concatenate([np.full(6,  0.2), np.full(3, 1.5)])

    print("Stage 2: fitting 6 misalign + scales ...")
    res2 = least_squares(resid_stage2, x0_stage2,
                         bounds=(lb2, ub2),
                         verbose=2,
                         loss='soft_l1',
                         x_scale='jac')

    cost2 = 0.5 * np.sum(res2.fun**2)
    print(" Stage2 cost:", cost2, " delta:", cost2 - cost1)

    gamma_opt = res2.x[:6]
    scale_opt = res2.x[6:9]

    out = {
        "success": bool(res2.success),
        "cost": cost2,
        "gamma": gamma_opt,
        "scale": scale_opt,
        "bias": bias_fixed
    }

    x_final = np.concatenate([gamma_opt, scale_opt, bias_fixed])
    return x_final, out

# =========================
# Accel points visualizer (unchanged)
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
# Compute segment pose vectors (a0,a1) and comparison logic
# =========================
def compute_segment_pose_vectors(accel_samples, gyro_segments, accel_params, before=10, after=10):
    idxs, vals = accel_samples_to_indexed_array(accel_samples)
    if idxs.size == 0:
        return []

    mean_norm = np.mean(np.linalg.norm(vals, axis=1))
    vals_ms2 = vals * g if 0.3 <= mean_norm <= 2 else vals
    vals_cal = accel_apply(vals_ms2, accel_params)

    poses = []
    for seg in gyro_segments:
        sidx = seg.get("start_accel_idx", None)
        eidx = seg.get("end_accel_idx", None)

        if sidx is None or eidx is None:
            continue

        # Average around the start/end accel sample indices
        a0 = average_accel_window(idxs, vals_cal, sidx, before=before, after=after).copy()
        a1 = average_accel_window(idxs, vals_cal, eidx, before=before, after=after).copy()

        n0 = np.linalg.norm(a0)
        n1 = np.linalg.norm(a1)
        if n0 < 1e-12 or n1 < 1e-12:
            continue

        a0 /= n0
        a1 /= n1
        poses.append((a0, a1))

    return poses

def integrate_segment_to_rotation_for_compare(seg, sample_rate, integrator="rk4", bias=None):
    seg_arr = np.array(seg, dtype=float)
    if seg_arr.size == 0:
        return np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3), 0.0

    # FIX: subtract bias, do not add
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
            return np.array([1.0, 0.0, 0.0, 0.0]), np.eye(3), 0.0
        R = rodrigues_exp(dtheta)
        return None, R, theta

    else:  # smallangle
        dtheta = np.sum(omega_r, axis=0) * dt
        K = skew(dtheta)
        R = np.eye(3) + K
        theta = np.linalg.norm(dtheta)
        return None, R, theta

def compare_tedaldi_style(
    accel_samples,
    gyro_segments,
    accel_params,
    sample_rate,
    integrator="rk4",
    bias=None,
    win_before=10,
    win_after=10,
    limit=200
):
    poses = compute_segment_pose_vectors(
        accel_samples,
        gyro_segments,
        accel_params,
        before=win_before,
        after=win_after
    )

    results = []
    n = min(len(poses), len(gyro_segments), limit)

    for i in range(n):
        a0, a1 = poses[i]
        seg = gyro_segments[i]["samples"]

        q, R, gyro_ang = integrate_segment_to_rotation_for_compare(
            seg,
            sample_rate,
            integrator=integrator,
            bias=bias
        )

        # predicted final gravity direction from gyro
        a1_pred = R @ a0
        a1_pred /= np.linalg.norm(a1_pred)

        # observable tilt change from accel
        accel_ang = angle_between_vectors_rad(a0, a1)

        # angle between gyro-predicted end direction and measured end direction
        pred_err = angle_between_vectors_rad(a1_pred, a1)

        results.append({
            "idx": i,
            "accel_angle_rad": accel_ang,
            "gyro_angle_rad": gyro_ang,
            "pred_error_rad": pred_err,
            "a0": a0,
            "a1": a1,
            "a1_pred": a1_pred
        })

        print(
            f"Seg {i:2d}: "
            f"accel_angle={np.degrees(accel_ang):7.3f} deg | "
            f"gyro_angle={np.degrees(gyro_ang):7.3f} deg | "
            f"pred_error={np.degrees(pred_err):7.3f} deg"
        )

    if results:
        errs_deg = np.array([np.degrees(r["pred_error_rad"]) for r in results])
        print("\nComparison summary:")
        print(f"  mean pred error   = {np.mean(errs_deg):.3f} deg")
        print(f"  median pred error = {np.median(errs_deg):.3f} deg")
        print(f"  max pred error    = {np.max(errs_deg):.3f} deg")
    else:
        print("No valid comparison segments found.")

    return results

# =========================
# Cross-calibration helper (uses parse_log_with_indices)
# =========================
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

def accel_angle(a0, a1):
    return angle_between_vectors_deg(a0, a1)

# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--accel-file", type=str, default=None,
                    help="Separate accelerometer-only log for sphere fit (optional)")
    ap.add_argument("--accel-params", type=str, default=None,
                    help="Either path to .npy/.txt file OR comma-separated list of 9 accel calibration params")
    ap.add_argument("--integrator", type=str, choices=["rk4","rodrigues","smallangle"], default="rk4",
                    help="Rotation integration method for gyro segments")
    ap.add_argument("--rate", type=float, default=None,
                    help="Override gyro sample rate (Hz)")
    ap.add_argument("--units", type=str, choices=["deg_s","rad_s"], default=None,
                    help="Gyro units in the log")

    args = ap.parse_args()

    # Apply overrides
    if args.rate is not None:
        globals()['SAMPLE_RATE_HZ'] = args.rate
    if args.units is not None:
        globals()['GYRO_UNITS'] = args.units
    if args.integrator:
        globals()['GYRO_INTEGRATOR'] = args.integrator

    # File handling
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

    # --- Parse main log with aligned indices ---
    accel_samples, gyro_segments, bias_from_log = parse_log_with_indices(path)

    # If a separate accelerometer-only file is provided, use its accel samples
    if args.accel_file:
        accel_samples_accel_file, _, _ = parse_log_with_indices(args.accel_file)
        if accel_samples_accel_file:
            accel_samples = accel_samples_accel_file

    # Load accel calibration parameters (file or inline list)
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

    # If we don't have accel samples at all -> exit
    idxs, accel_vals = accel_samples_to_indexed_array(accel_samples)
    if accel_vals.size == 0:
        sys.exit("No accel data found in the parsed logs.")

    # Convert accel raw samples to m/s^2 when appropriate (same heuristic as before)
    mean_norm = np.mean(np.linalg.norm(accel_vals, axis=1))
    accel_raw_all = accel_vals * g if 0.3 <= mean_norm <= 2 else accel_vals

    # If acc_p not provided, fit using all accel samples (sphere fit)
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

    # Optional cross-calibration test (keeps original behaviour)
    compare_cross_calibration("DATA/big_turntable_test_sample.txt", acc_p)

    # --- Tedaldi-style comparisons (project onto plane orthogonal to gravity) ---
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

    # Build accel_cal array for gyro calibration: take first a0 and then each a1
    poses = compute_segment_pose_vectors(accel_samples, gyro_segments, acc_p)
    if poses:
        # Construct accel_cal with length = (#segments) + 1
        accel_cal_list = []
        # first pose a0 of first segment
        accel_cal_list.append(poses[0][0])
        for (a0, a1) in poses:
            accel_cal_list.append(a1)
        accel_cal = np.vstack(accel_cal_list)  # shape (n_segments+1, 3)
    else:
        accel_cal = accel_cal_all  # fallback: use calibrated sample cloud (may not be correct format)

    # Use gyro_segments (parsed earlier) as gyros for calibration (samples arrays)
    gyros = [seg["samples"] for seg in gyro_segments]

    # =========================
    # Gyroscope calibration
    # =========================
    if gyros:
        print(f"\nFound {len(gyros)} gyro segments.")
        # Ensure accel_cal and gyro segment count agree; trim if needed
        if len(gyros) != len(accel_cal) - 1:
            n = min(len(gyros), len(accel_cal) - 1)
            gyros = gyros[:n]
            accel_cal = accel_cal[:n + 1]

        # ---- FILTER OUT STATIC OR NEAR-STATIC SEGMENTS ----
        print("Filtering gyro segments for real motion...")

        static_bias = bias_from_log if bias_from_log is not None and getattr(bias_from_log, "size", 0) == 3 else None
        if static_bias is not None:
            bias_est = static_bias
        else:
            all_samples = np.vstack([seg for seg in gyros if seg.size > 0]) if gyros else np.zeros((0,3))
            bias_est = np.mean(all_samples, axis=0) if all_samples.size else np.zeros(3)
        print(f"Estimated gyro bias for filtering: {bias_est}")

        # 3. Calibrate
        _, gyro_info = calibrate_gyro(accel_cal, gyros, static_bias)
        print("\n--- Per-segment Z scale test ---")

        # Reconstruct dirs used in calibration
        dirs = accel_cal / np.linalg.norm(accel_cal, axis=1, keepdims=True)
        

        # Get fitted values
        scale_fit = gyro_info["scale"]
        bias_fit = gyro_info["bias"]

        for candidate in [0.96, 1.0, 1.02]:
            test_scales = np.array([scale_fit[0], scale_fit[1], candidate])
            errs = per_segment_errors(test_scales, bias_fit, dirs, gyros)
            print("candidate", candidate,
                "median err:", np.nanmedian(errs),
                "mean err:", np.nanmean(errs))

        print("\nGyro calibration results:")
        if "gamma" in gyro_info:
            print(" Misalign gamma:", gyro_info["gamma"])
        print(" Scale:", gyro_info["scale"])
        print(" Bias:", gyro_info["bias"])
        print(" Success:", gyro_info["success"], " Cost:", gyro_info["cost"])
        print("\nGyro calibration results (one row):")

        # misalignment padded to 6 values
        gamma = gyro_info.get("gamma", np.zeros(6))
        one_row = np.concatenate([bias_fit, scale_fit, gamma])
        print(" ".join(f"{x:.8e}" for x in one_row))
    else:
        print("No gyro segments found.")