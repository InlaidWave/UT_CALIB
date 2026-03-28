import os
import re
import sys
import argparse
import numpy as np
from scipy.optimize import least_squares

# =========================
# Configuration
# =========================
DATA_FOLDER = "DATA"
g = 9.81872
SAMPLE_RATE_HZ = 70.0
GYRO_UNITS = "deg_s"          # "deg_s" or "rad_s"
GYRO_AXIS_SIGN = np.array([1.0, 1.0, 1.0], dtype=float)
GYRO_INTEGRATOR = "rk4"       # this script uses rk4 for calibration/debug

# =========================
# Regex patterns
# =========================
ACCEL_PAT = re.compile(
    r"(?:A?X)\s*([-+]?\d*\.?\d+)\s*(?:A?Y|Y)\s*([-+]?\d*\.?\d+)\s*(?:A?Z|Z)\s*([-+]?\d*\.?\d+)",
    re.I
)
GYRO_PAT = re.compile(
    r"GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)",
    re.I
)
GYRO_BIAS_PAT = re.compile(
    r"GYRO_BIAS.*?GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)",
    re.I
)
MODE_START_PAT = re.compile(r"MODE\s*=\s*GYRO_START", re.I)
MODE_END_PAT = re.compile(r"MODE\s*=\s*GYRO_END", re.I)


# =========================
# Basic helpers
# =========================
def list_txt_files(directory):
    folder_path = os.path.join(directory, DATA_FOLDER)
    if not os.path.isdir(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".txt")]


def unit_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def angle_between_vectors_rad(v1, v2, eps=1e-12):
    u1 = unit_vector(v1, eps=eps)
    u2 = unit_vector(v2, eps=eps)
    c = float(np.dot(u1, u2))
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))


def angle_between_vectors_deg(v1, v2, eps=1e-12):
    return float(np.degrees(angle_between_vectors_rad(v1, v2, eps=eps)))


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y,  x,  0.0]], dtype=float)


def rodrigues_exp(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + skew(phi)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def build_Ta(a_yz, a_zy, a_zx):
    return np.array([[1.0, -a_yz,  a_zy],
                     [0.0,  1.0,  -a_zx],
                     [0.0,  0.0,   1.0]], dtype=float)


def build_K(sx, sy, sz):
    return np.diag([sx, sy, sz])


def build_Tg(g_yz, g_zy, g_xz, g_zx, g_xy, g_yx):
    return np.array([[1.0,   -g_yz,  g_zy],
                     [g_xz,   1.0,   -g_zx],
                     [-g_xy,  g_yx,   1.0]], dtype=float)


def omega_to_rad_s(omega):
    omega = np.asarray(omega, dtype=float)
    return np.deg2rad(omega) if GYRO_UNITS.lower() == "deg_s" else omega

# =========================
# Quaternion / RK4
# =========================
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
    ], dtype=float)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z),   2.0*(x*y - z*w),         2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),         1.0 - 2.0*(x*x + z*z),   2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),         2.0*(y*z + x*w),         1.0 - 2.0*(x*x + y*y)],
    ], dtype=float)


def rk4n_integrate_quat(omegas_rad_s, dt):
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    for w in omegas_rad_s:
        def f(qv):
            omega_q = np.array([0.0, w[0], w[1], w[2]], dtype=float)
            return 0.5 * quat_multiply(qv, omega_q)

        k1 = f(q)
        k2 = f(q + 0.5 * dt * k1)
        k3 = f(q + 0.5 * dt * k2)
        k4 = f(q + dt * k3)
        q = q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        q = quat_normalize(q)

    return q


def integrate_gyro_segment_rk4(seg, dt, bias=None, scales=None, gamma=None):
    seg = np.asarray(seg, dtype=float)
    if seg.size == 0:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q, np.eye(3), 0.0, 0.0

    if bias is None:
        bias = np.zeros(3, dtype=float)
    if scales is None:
        scales = np.ones(3, dtype=float)
    if gamma is None:
        gamma = np.zeros(6, dtype=float)

    bias = np.asarray(bias, dtype=float)
    scales = np.asarray(scales, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    raw = (seg - bias[None, :]) * GYRO_AXIS_SIGN[None, :]
    Kg = np.diag(scales)
    Tg = build_Tg(*gamma)

    omega_corr = (Tg @ (Kg @ raw.T)).T
    omega_rad = omega_to_rad_s(omega_corr)

    q = rk4n_integrate_quat(omega_rad, dt)
    R = quat_to_R(q)

    # Principal final-angle from quaternion (0..180 deg)
    w = np.clip(q[0], -1.0, 1.0)
    quat_angle = 2.0 * np.arccos(w)

    # Path-angle proxy = || sum(omega*dt) ||, can exceed 180 deg
    vecsum_angle = np.linalg.norm(np.sum(omega_rad, axis=0) * dt)

    return q, R, quat_angle, vecsum_angle


# =========================
# Parsing
# =========================
def parse_accel_line(line):
    m = ACCEL_PAT.search(line.strip())
    if not m:
        return None
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=float)


def parse_gyro_line(line):
    m = GYRO_PAT.search(line.strip())
    if not m:
        return None
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=float)


def parse_log(path):
    accel_samples = []
    gyro_segments = []
    bias = None

    in_seg = False
    seg_buf = []
    seg_start_line_no = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            m_bias = GYRO_BIAS_PAT.search(line)
            if m_bias:
                bias = np.array([float(m_bias.group(1)), float(m_bias.group(2)), float(m_bias.group(3))], dtype=float)
                continue

            if MODE_START_PAT.search(line):
                in_seg = True
                seg_buf = []
                seg_start_line_no = line_no
                continue

            if MODE_END_PAT.search(line):
                if in_seg:
                    gyro_segments.append({
                        "start_line_no": seg_start_line_no,
                        "end_line_no": line_no,
                        "samples": np.array(seg_buf, dtype=float) if seg_buf else np.zeros((0, 3), dtype=float),
                    })
                in_seg = False
                seg_buf = []
                seg_start_line_no = None
                continue

            gline = parse_gyro_line(line)
            if gline is not None:
                if in_seg:
                    seg_buf.append(gline)
                continue

            aline = parse_accel_line(line)
            if aline is not None:
                accel_samples.append({
                    "line_no": line_no,
                    "raw": aline,
                    "text": line,
                })
                continue

    return accel_samples, gyro_segments, bias


def accel_samples_to_indexed_array(accel_samples):
    if not accel_samples:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=float)

    idxs = np.array([entry["line_no"] for entry in accel_samples], dtype=int)
    vals = np.vstack([entry["raw"] for entry in accel_samples]).astype(float)
    return idxs, vals


def find_prev_accel(accel_samples, line_no):
    prev_entry = None
    for entry in accel_samples:
        if entry["line_no"] < line_no:
            prev_entry = entry
        else:
            break
    return prev_entry


def find_next_accel(accel_samples, line_no):
    for entry in accel_samples:
        if entry["line_no"] > line_no:
            return entry
    return None


# =========================
# Accelerometer calibration
# =========================
def maybe_convert_accel_to_ms2(vals):
    if vals.size == 0:
        return vals
    mean_norm = float(np.mean(np.linalg.norm(vals, axis=1)))
    return vals * g if 0.3 <= mean_norm <= 2.0 else vals


def accel_resid(p, data):
    bx, by, bz, sx, sy, sz, a_yz, a_zy, a_zx = p
    b = np.array([bx, by, bz], dtype=float)
    K = build_K(sx, sy, sz)
    T = build_Ta(a_yz, a_zy, a_zx)
    corr = (T @ (K @ (data.T - b[:, None]))).T
    return np.linalg.norm(corr, axis=1) - g


def accel_find_params(data):
    init = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    res = least_squares(accel_resid, init, args=(data,), loss="soft_l1", x_scale="jac")
    return res.x


def accel_apply(data, p):
    bx, by, bz, sx, sy, sz, a_yz, a_zy, a_zx = p
    b = np.array([bx, by, bz], dtype=float)
    K = build_K(sx, sy, sz)
    T = build_Ta(a_yz, a_zy, a_zx)
    return (T @ (K @ (data.T - b[:, None]))).T


def accel_summary(calib_ms2, raw_ms2):
    m1 = np.linalg.norm(calib_ms2, axis=1)
    m2 = np.linalg.norm(raw_ms2, axis=1)
    r1 = m1 - g
    r2 = m2 - g
    print(f"Accel: mean offset from g (calib): {np.mean(r1):.6f}")
    print(f"Accel: mean offset from g (raw):   {np.mean(r2):.6f}")
    print(f"Accel: RMS fit to sphere (calib): {np.sqrt(np.mean(r1**2)):.6f}")
    print(f"Accel: RMS fit to sphere (raw):   {np.sqrt(np.mean(r2**2)):.6f}")


# =========================
# Pose extraction for gyro segments
# =========================
def build_calibrated_accel_lookup(accel_samples, accel_params):
    idxs, vals = accel_samples_to_indexed_array(accel_samples)
    vals_ms2 = maybe_convert_accel_to_ms2(vals)
    vals_cal = accel_apply(vals_ms2, accel_params)

    calib_by_line = {}
    for i, entry in enumerate(accel_samples):
        calib_by_line[entry["line_no"]] = vals_cal[i]

    return calib_by_line


def compute_segment_pose_vectors(accel_samples, gyro_segments, accel_params):
    calib_by_line = build_calibrated_accel_lookup(accel_samples, accel_params)

    poses = []
    for i, seg in enumerate(gyro_segments):
        start_line = seg["start_line_no"]
        end_line = seg["end_line_no"]

        a0_entry = find_prev_accel(accel_samples, start_line)
        a1_entry = find_next_accel(accel_samples, end_line)

        if a0_entry is None or a1_entry is None:
            continue

        a0_raw = calib_by_line[a0_entry["line_no"]].copy()
        a1_raw = calib_by_line[a1_entry["line_no"]].copy()

        poses.append({
            "segment_index": i,
            "start_accel_line": a0_entry["line_no"],
            "end_accel_line": a1_entry["line_no"],
            "a0_raw": a0_raw,
            "a1_raw": a1_raw,
            "a0": unit_vector(a0_raw),
            "a1": unit_vector(a1_raw),
            "gyro_samples": seg["samples"],
        })

    return poses


# =========================
# Gyro calibration (RK4-based)
# =========================
def calibrate_gyro_rk4(poses, static_bias=None, verbose=True):
    if not poses:
        raise ValueError("No pose pairs available for gyro calibration.")

    gyros = [p["gyro_samples"] for p in poses]

    if static_bias is not None and getattr(static_bias, "size", 0) == 3:
        bias_fixed = np.asarray(static_bias, dtype=float)
    else:
        all_samples = np.vstack([seg for seg in gyros if seg.size > 0]) if gyros else np.zeros((0, 3), dtype=float)
        bias_fixed = np.mean(all_samples, axis=0) if all_samples.size else np.zeros(3, dtype=float)

    dt = 1.0 / SAMPLE_RATE_HZ

    def motion_residual(params):
        gamma = params[:6]
        scales = params[6:9]

        res_list = []
        for pose in poses:
            a0 = pose["a0"]
            a1 = pose["a1"]
            seg = pose["gyro_samples"]
            if seg.size == 0:
                continue

            _, R, _, _ = integrate_gyro_segment_rk4(
                seg,
                dt=dt,
                bias=bias_fixed,
                scales=scales,
                gamma=gamma,
            )

            res_list.append((R.T @ a0) - a1)

        motion = np.concatenate(res_list) if res_list else np.zeros(0, dtype=float)

        # weak regularization on gamma and scale drift
        gamma_reg = gamma / 2
        scale_reg = (scales - 1.0) / 1

        return np.concatenate([motion, gamma_reg, scale_reg])

    # Stage 1: fit only scales
    def resid_stage1(scales):
        params = np.concatenate([np.zeros(6, dtype=float), scales])
        return motion_residual(params)

    x0_stage1 = np.ones(3, dtype=float)
    lb1 = np.full(3, 0.5, dtype=float)
    ub1 = np.full(3, 1.5, dtype=float)

    if verbose:
        print("\nStage 1: fitting gyro scales with RK4 ...")

    res1 = least_squares(
        resid_stage1,
        x0_stage1,
        bounds=(lb1, ub1),
        loss="soft_l1",
        x_scale="jac",
        verbose=2 if verbose else 0,
    )

    scales1 = res1.x.copy()
    cost1 = 0.5 * np.sum(res1.fun ** 2)

    # Stage 2: fit 6 misalignment + 3 scales
    x0_stage2 = np.concatenate([np.zeros(6, dtype=float), scales1])
    lb2 = np.concatenate([np.full(6, -0.2), np.full(3, 0.5)])
    ub2 = np.concatenate([np.full(6,  0.2), np.full(3, 1.5)])

    if verbose:
        print("\nStage 2: fitting gyro misalignment + scales with RK4 ...")

    res2 = least_squares(
        motion_residual,
        x0_stage2,
        bounds=(lb2, ub2),
        loss="soft_l1",
        x_scale="jac",
        verbose=2 if verbose else 0,
    )

    gamma_opt = res2.x[:6]
    scales_opt = res2.x[6:9]
    cost2 = 0.5 * np.sum(res2.fun ** 2)

    out = {
        "success": bool(res2.success),
        "cost_stage1": float(cost1),
        "cost_stage2": float(cost2),
        "gamma": gamma_opt,
        "scale": scales_opt,
        "bias": bias_fixed,
        "message": res2.message,
    }
    return out


# =========================
# Debugging / comparison
# =========================
def debug_consecutive_accel_angles(accel_samples, accel_params):
    print("\n=== Consecutive calibrated accel angles ===")
    if len(accel_samples) < 2:
        print("Not enough accel samples.")
        return

    calib_by_line = build_calibrated_accel_lookup(accel_samples, accel_params)

    for i in range(len(accel_samples) - 1):
        e0 = accel_samples[i]
        e1 = accel_samples[i + 1]
        v0 = calib_by_line[e0["line_no"]]
        v1 = calib_by_line[e1["line_no"]]
        ang = angle_between_vectors_deg(v0, v1)
        print(
            f"accel line {e0['line_no']:4d} -> line {e1['line_no']:4d}: {ang:9.6f} deg"
        )


def debug_segment_angles(poses, bias, scales, gamma, title):
    print(f"\n=== {title} ===")
    if not poses:
        print("No valid segment pose pairs.")
        return

    dt = 1.0 / SAMPLE_RATE_HZ
    pred_errs = []

    for pose in poses:
        seg_idx = pose["segment_index"]
        a0 = pose["a0"]
        a1 = pose["a1"]
        seg = pose["gyro_samples"]

        q, R, quat_ang, vecsum_ang = integrate_gyro_segment_rk4(
            seg,
            dt=dt,
            bias=bias,
            scales=scales,
            gamma=gamma,
        )

        a1_pred = unit_vector(R.T @ a0)
        accel_ang = angle_between_vectors_deg(a0, a1)
        pred_err = angle_between_vectors_deg(a1_pred, a1)
        pred_errs.append(pred_err)

        print(f"\nSegment {seg_idx}")
        print(f"  start accel line: {pose['start_accel_line']}")
        print(f"  end   accel line: {pose['end_accel_line']}")
        print(f"  a0 raw : {pose['a0_raw']}")
        print(f"  a1 raw : {pose['a1_raw']}")
        print(f"  a0 unit: {np.array2string(a0, precision=6, suppress_small=False)}")
        print(f"  a1 unit: {np.array2string(a1, precision=6, suppress_small=False)}")
        print(f"  accel angle               : {accel_ang:9.6f} deg")
        print(f"  gyro integrated angle(q)  : {np.degrees(quat_ang):9.6f} deg")
        print(f"  gyro path angle(sum wdT)  : {np.degrees(vecsum_ang):9.6f} deg")
        print(f"  end-vector prediction err : {pred_err:9.6f} deg")
        print(f"  gyro samples in segment   : {len(seg)}")

    if pred_errs:
        pred_errs = np.array(pred_errs, dtype=float)
        print("\nPrediction error summary:")
        print(f"  mean   : {np.mean(pred_errs):.6f} deg")
        print(f"  median : {np.median(pred_errs):.6f} deg")
        print(f"  max    : {np.max(pred_errs):.6f} deg")


def debug_two_lines(line1, line2):
    v1 = parse_accel_line(line1)
    v2 = parse_accel_line(line2)
    if v1 is None or v2 is None:
        raise ValueError("Could not parse one of the provided accel lines.")

    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    ang = angle_between_vectors_deg(v1, v2)

    print("\n=== Direct two-line debug ===")
    print("line1:", line1)
    print("line2:", line2)
    print("v1:", v1)
    print("v2:", v2)
    print("u1:", np.array2string(u1, precision=6, suppress_small=False))
    print("u2:", np.array2string(u2, precision=6, suppress_small=False))
    print(f"angle: {ang:.6f} deg")

def calibrate_gyro_one_stage(accel_calib, gyro_segments, static_bias=None,
                                         use_rodrigues=True, subsample=None,
                                         max_nfev=500, tol=1e-8,
                                         print_every=5, verbose=True):
    """
    Single-stage LM fit (method='lm') with continuous cost reporting.
    Returns: x_opt (9,), out dict with keys including cost_initial, cost, cost_history (list)
    Params:
      - accel_calib: (N+1,3) accel direction vectors (will be normalized)
      - gyro_segments: list of Nx3 arrays or dicts with "samples"
      - static_bias: optional fixed bias (3,)
      - use_rodrigues: if True use summed-angle integrator during fit (faster)
      - subsample: integer >=2 reduces data per-segment (speeds up)
      - max_nfev: maximum function evaluations (prevents very long runs)
      - tol: solver tolerances (applied to xtol/ftol/gtol)
      - print_every: print cost every N residual evaluations
    """
    acc = np.asarray(accel_calib, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("accel_calib must have shape (N+1,3)")

    # normalize accel directions
    norms = np.linalg.norm(acc, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("accel_calib contains zero (or near-zero) vector")
    acc = (acc.T / norms).T

    # prepare gyros list (subsample optionally)
    gyros = []
    for seg in gyro_segments:
        s = np.asarray(seg["samples"] if isinstance(seg, dict) and "samples" in seg else seg, dtype=float)
        if subsample and int(subsample) > 1 and s.shape[0] > 1:
            s = s[::int(subsample)]
        gyros.append(s)

    # fixed bias
    if static_bias is not None and getattr(static_bias, "size", 0) == 3:
        bias_fixed = np.array(static_bias, dtype=float)
    else:
        all_samples = np.vstack([s for s in gyros if s.size > 0]) if gyros else np.zeros((0,3))
        bias_fixed = np.mean(all_samples, axis=0) if all_samples.size else np.zeros(3, dtype=float)
    if verbose:
        print("Using fixed gyro bias (for residuals):", bias_fixed)
        print(f"LM fit settings: use_rodrigues={use_rodrigues}, subsample={subsample}, max_nfev={max_nfev}")

    dt = 1.0 / SAMPLE_RATE_HZ

    # helper: build matrices from params
    def params_to_matrices(x):
        g_yz, g_zy, g_xz, g_zx, g_xy, g_yx = x[0:6]
        sgx, sgy, sgz = x[6:9]
        Tg = build_Tg(g_yz, g_zy, g_xz, g_zx, g_xy, g_yx)
        Kg = np.diag([sgx, sgy, sgz])
        return Tg, Kg

    # precompute corrected segments for speed (bias subtracted)
    seg_summaries = []
    for s in gyros:
        if s.size == 0:
            seg_summaries.append(None)
            continue
        seg_corr = s - bias_fixed[None, :]
        seg_summaries.append({"seg_corr": seg_corr, "n": seg_corr.shape[0]})

    # bookkeeping for progress
    cost_history = []
    eval_count = {"n": 0}  # mutable counter closed over residuals

    # residual function wrapper that also records cost
    def residuals_and_record(x):
        Tg, Kg = params_to_matrices(x)
        res_list = []
        for k in range(1, len(acc)):
            a0 = acc[k-1]
            a1 = acc[k]
            summary = seg_summaries[k-1] if (k-1) < len(seg_summaries) else None
            if summary is None:
                continue
            seg_corr = summary["seg_corr"]  # (M,3)
            if seg_corr.size == 0:
                continue

            omega = (Tg @ (Kg @ seg_corr.T)).T

            if use_rodrigues:
                omega_r = omega_to_rad_s(omega)
                dtheta = np.sum(omega_r, axis=0) * dt1
                theta = np.linalg.norm(dtheta)
                if theta < 1e-12:
                    R = np.eye(3)
                else:
                    k_axis = dtheta / theta
                    K = skew(k_axis)
                    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            else:
                omega_r = omega_to_rad_s(omega)
                qf = rk4n_integrate_quat(omega_r, dt)
                R = quat_to_R(qf)

            rvec = (R @ a0) - a1
            res_list.append(rvec)

        if not res_list:
            res = np.zeros(0, dtype=float)
        else:
            res = np.concatenate(res_list, axis=0)

        # record cost and optionally print
        cost = 0.5 * float(np.sum(res**2))
        cost_history.append(cost)
        eval_count["n"] += 1
        if verbose and (eval_count["n"] % print_every == 0 or eval_count["n"] == 1):
            print(f"[LM eval {eval_count['n']}] cost = {cost:.6e}")

        return res

    # initial guess
    x0 = np.concatenate([np.zeros(6, dtype=float), np.ones(3, dtype=float)])
    f0 = residuals_and_record(x0)
    cost_initial = 0.5 * float(np.sum(f0**2))
    if verbose:
        print("Initial cost:", cost_initial)

    # run LM: use least_squares(method='lm'), allow max_nfev
    res = least_squares(residuals_and_record, x0, method='lm',
                        xtol=tol, ftol=tol, gtol=tol,
                        max_nfev=max_nfev, verbose=2)

    x_opt = res.x.copy()
    final_cost = 0.5 * float(np.sum(res.fun**2))

    out = {
        "success": bool(res.success),
        "cost_initial": float(cost_initial),
        "cost": float(final_cost),
        "gamma": x_opt[0:6],
        "scale": x_opt[6:9],
        "bias": bias_fixed,
        "message": res.message,
        "nfev": int(res.nfev),
        "njev": getattr(res, "njev", None),
        "cost_history": cost_history,
        "eval_count": eval_count["n"]
    }

    if verbose:
        print("LM finished: success=", out["success"], "nfev=", out["nfev"])
        print(" initial cost:", out["cost_initial"], " final cost:", out["cost"])
        print(" gamma:", out["gamma"])
        print(" scale:", out["scale"])

    return x_opt, out
# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None, help="Main log file with accel + gyro segments")
    ap.add_argument("--accel-file", type=str, default=None, help="Separate accel-only log used to fit accelerometer parameters")
    ap.add_argument("--accel-params", type=str, default=None,
                    help="Optional: path to .npy/.txt OR comma-separated 9 accel params")
    ap.add_argument("--rate", type=float, default=None, help="Override sample rate")
    ap.add_argument("--units", type=str, choices=["deg_s", "rad_s"], default=None, help="Gyro units")
    ap.add_argument("--debug-line1", type=str, default=None, help="Direct accel line 1")
    ap.add_argument("--debug-line2", type=str, default=None, help="Direct accel line 2")
    args = ap.parse_args()

    if args.debug_line1 is not None and args.debug_line2 is not None:
        debug_two_lines(args.debug_line1, args.debug_line2)
        sys.exit(0)

    if args.rate is not None:
        SAMPLE_RATE_HZ = float(args.rate)
    if args.units is not None:
        GYRO_UNITS = args.units

    if args.file is not None:
        main_path = args.file
    else:
        txts = list_txt_files(".")
        if not txts:
            sys.exit("No .txt files found.")
        if len(txts) == 1:
            main_path = os.path.join(DATA_FOLDER, txts[0])
        else:
            for i, f in enumerate(txts):
                print(f"{i}: {f}")
            sel = int(input("Select: "))
            main_path = os.path.join(DATA_FOLDER, txts[sel])

    main_accel_samples, gyro_segments, bias_from_log = parse_log(main_path)

    print(f"\nLoaded main file: {main_path}")
    print(f"Main accel entries: {len(main_accel_samples)}")
    print(f"Gyro segments:      {len(gyro_segments)}")
    print(f"Bias from log:      {bias_from_log}")

    if not main_accel_samples:
        sys.exit("No accel entries found in main file.")
    if not gyro_segments:
        sys.exit("No gyro segments found in main file.")

    # Decide which accel data to use for accel parameter fit
    accel_fit_samples = main_accel_samples
    if args.accel_file:
        accel_fit_samples, _, _ = parse_log(args.accel_file)
        print(f"Loaded accel-fit file: {args.accel_file}")
        print(f"Accel-fit entries:     {len(accel_fit_samples)}")
        if not accel_fit_samples:
            sys.exit("No accel entries found in --accel-file.")

    # Get accel calibration parameters
    acc_p = None
    if args.accel_params:
        try:
            if os.path.isfile(args.accel_params):
                if args.accel_params.endswith(".npy"):
                    acc_p = np.load(args.accel_params)
                else:
                    acc_p = np.loadtxt(args.accel_params)
            else:
                values = [float(v.strip()) for v in args.accel_params.split(",")]
                if len(values) != 9:
                    raise ValueError("Expected 9 params: bx,by,bz,sx,sy,sz,a_yz,a_zy,a_zx")
                acc_p = np.array(values, dtype=float)
        except Exception as e:
            sys.exit(f"Failed to load --accel-params: {e}")
    else:
        _, accel_fit_vals = accel_samples_to_indexed_array(accel_fit_samples)
        accel_fit_raw_ms2 = maybe_convert_accel_to_ms2(accel_fit_vals)
        acc_p = accel_find_params(accel_fit_raw_ms2)

    print("\nAccelerometer calibration parameters:")
    print("  Bias:", acc_p[:3])
    print("  Scale:", acc_p[3:6])
    print("  Misalignment (a_yz, a_zy, a_zx):", acc_p[6:])
    print("  One row:", " ".join(f"{x:.8e}" for x in acc_p))

    # Summary of accel calibration quality on fit file
    _, accel_fit_vals = accel_samples_to_indexed_array(accel_fit_samples)
    accel_fit_raw_ms2 = maybe_convert_accel_to_ms2(accel_fit_vals)
    accel_fit_cal_ms2 = accel_apply(accel_fit_raw_ms2, acc_p)
    print("\nAccelerometer fit summary:")
    accel_summary(accel_fit_cal_ms2, accel_fit_raw_ms2)

    # Debug consecutive accel angles on main file
    debug_consecutive_accel_angles(main_accel_samples, acc_p)

    # Build pose pairs from main file
    poses = compute_segment_pose_vectors(main_accel_samples, gyro_segments, acc_p)
    poses = [p for p in poses if p["segment_index"] != 4]
    print(f"\nValid segment pose pairs: {len(poses)}")
    if not poses:
        sys.exit("Could not form any valid segment pose pairs.")

    # Debug before gyro calibration
    pre_bias = bias_from_log if bias_from_log is not None else np.zeros(3, dtype=float)
    debug_segment_angles(
        poses=poses,
        bias=pre_bias,
        scales=np.ones(3, dtype=float),
        gamma=np.zeros(6, dtype=float),
        title="Before gyro capython adflsh2.py --file DATA/gyro_clean_movements.txt --accel-file DATA/TURNTABLE1_ACCEL.txtlibration",
    )

    # Gyro calibration using RK4
    accel_dirs = np.vstack([poses[0]["a0"]] + [p["a1"] for p in poses])

    gyros = [p["gyro_samples"] for p in poses]

#     x_opt, gyro_info = calibrate_gyro_one_stage(  
#         accel_dirs, gyros,
#         static_bias=bias_from_log,
#         use_rodrigues=True,     # faster integrator for debug
#         subsample=2,            # optional
#         max_nfev=2000,          # keep runs bounded
#         print_every=2,
#         verbose=True
# )

    gyro_info = calibrate_gyro_rk4(
        poses,
        static_bias=bias_from_log,
        verbose=True
    )

    print("\nGyro calibration results:")
    print("  Success:", gyro_info["success"])
    print("  Message:", gyro_info["message"])
    print("  Cost stage1:", gyro_info["cost_stage1"])
    print("  Cost stage2:", gyro_info["cost_stage2"])
    print("  Bias:", gyro_info["bias"])
    print("  Scale:", gyro_info["scale"])
    print("  Gamma:", gyro_info["gamma"])

    one_row = np.concatenate([gyro_info["bias"], gyro_info["scale"], gyro_info["gamma"]])
    print("\nGyro calibration results (one row):")
    print(" ".join(f"{x:.8e}" for x in one_row))

    # Debug after gyro calibration
    debug_segment_angles(
        poses=poses,
        bias=gyro_info["bias"],
        scales=gyro_info["scale"],
        gamma=gyro_info["gamma"],
        title="After gyro calibration",
    )


