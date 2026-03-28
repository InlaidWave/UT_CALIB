import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import time
import os
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


def format_standard(value, decimals=6):
    text = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def format_scientific(value, decimals=6):
    return f"{float(value):.{decimals}e}"


def format_scientific_sequence(values, decimals=6):
    return " ".join(format_scientific(value, decimals) for value in values)


def format_matrix_block(name, matrix, decimals=6):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = [" ".join(format_scientific(value, decimals) for value in row) for row in arr]
    return f"{name} =\n" + "\n".join(rows)


def format_named_values(names, values, decimals=6):
    return ", ".join(
        f"{name}={format_standard(value, decimals)}"
        for name, value in zip(names, values)
    )


def build_sensor_parameter_rows(result):
    accel = result["accel_params"]
    gyro = result["gyro_info"]

    rows = [
        {"sensor": "accel", "group": "bias", "parameter": "x", "value": float(accel["bias_x"])},
        {"sensor": "accel", "group": "bias", "parameter": "y", "value": float(accel["bias_y"])},
        {"sensor": "accel", "group": "bias", "parameter": "z", "value": float(accel["bias_z"])},
        {"sensor": "accel", "group": "scale", "parameter": "x", "value": float(accel["scale_x"])},
        {"sensor": "accel", "group": "scale", "parameter": "y", "value": float(accel["scale_y"])},
        {"sensor": "accel", "group": "scale", "parameter": "z", "value": float(accel["scale_z"])},
        {"sensor": "accel", "group": "misalignment", "parameter": "a_yz", "value": float(accel["a_yz"])},
        {"sensor": "accel", "group": "misalignment", "parameter": "a_zy", "value": float(accel["a_zy"])},
        {"sensor": "accel", "group": "misalignment", "parameter": "a_zx", "value": float(accel["a_zx"])},
        {"sensor": "gyro", "group": "bias", "parameter": "x", "value": float(gyro["bias"][0])},
        {"sensor": "gyro", "group": "bias", "parameter": "y", "value": float(gyro["bias"][1])},
        {"sensor": "gyro", "group": "bias", "parameter": "z", "value": float(gyro["bias"][2])},
        {"sensor": "gyro", "group": "scale", "parameter": "x", "value": float(gyro["scale"][0])},
        {"sensor": "gyro", "group": "scale", "parameter": "y", "value": float(gyro["scale"][1])},
        {"sensor": "gyro", "group": "scale", "parameter": "z", "value": float(gyro["scale"][2])},
        {"sensor": "gyro", "group": "misalignment", "parameter": "g_yz", "value": float(gyro["gamma"][0])},
        {"sensor": "gyro", "group": "misalignment", "parameter": "g_zy", "value": float(gyro["gamma"][1])},
        {"sensor": "gyro", "group": "misalignment", "parameter": "g_xz", "value": float(gyro["gamma"][2])},
        {"sensor": "gyro", "group": "misalignment", "parameter": "g_zx", "value": float(gyro["gamma"][3])},
        {"sensor": "gyro", "group": "misalignment", "parameter": "g_xy", "value": float(gyro["gamma"][4])},
        {"sensor": "gyro", "group": "misalignment", "parameter": "g_yx", "value": float(gyro["gamma"][5])},
    ]
    return rows

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
        return np.concatenate([motion])

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

def run_pipeline(main_path, accel_file_path=None, accel_params_arg=None, sample_rate=None, units=None, verbose=False):
    global SAMPLE_RATE_HZ, GYRO_UNITS
    if sample_rate:
        SAMPLE_RATE_HZ = float(sample_rate)
    if units:
        GYRO_UNITS = units

    main_accel_samples, gyro_segments, bias_from_log = parse_log(main_path)

    if not main_accel_samples:
        raise RuntimeError("No accel entries found in main file.")
    if not gyro_segments:
        raise RuntimeError("No gyro segments found in main file.")

    # select accel-fit
    if accel_file_path:
        accel_fit_samples, _, _ = parse_log(accel_file_path)
        if not accel_fit_samples:
            raise RuntimeError("No accel entries found in accel-file.")
    else:
        accel_fit_samples = main_accel_samples

    # get accel params array
    if accel_params_arg is not None:
        if isinstance(accel_params_arg, str) and os.path.isfile(accel_params_arg):
            if accel_params_arg.endswith(".npy"):
                acc_p_arr = np.load(accel_params_arg)
            else:
                acc_p_arr = np.loadtxt(accel_params_arg)
        elif isinstance(accel_params_arg, (list, tuple, np.ndarray)):
            acc_p_arr = np.asarray(accel_params_arg, dtype=float)
        else:
            try:
                parts = [float(x.strip()) for x in str(accel_params_arg).split(",")]
                acc_p_arr = np.array(parts, dtype=float)
            except Exception:
                raise RuntimeError("Could not interpret accel_params_arg")
    else:
        _, accel_fit_vals = accel_samples_to_indexed_array(accel_fit_samples)
        accel_fit_raw_ms2 = maybe_convert_accel_to_ms2(accel_fit_vals)
        acc_p_arr = accel_find_params(accel_fit_raw_ms2)

    # compute accel fit summary
    _, accel_fit_vals = accel_samples_to_indexed_array(accel_fit_samples)
    accel_fit_raw_ms2 = maybe_convert_accel_to_ms2(accel_fit_vals)
    accel_fit_cal_ms2 = accel_apply(accel_fit_raw_ms2, acc_p_arr)

    # poses used for gyro calibration
    poses = compute_segment_pose_vectors(main_accel_samples, gyro_segments, acc_p_arr)

    dt = 1.0 / SAMPLE_RATE_HZ
    scored_poses = []

    for pose in poses:
        seg = pose["gyro_samples"]

        q, R, quat_ang, _ = integrate_gyro_segment_rk4(
            seg,
            dt=dt,
            bias=bias_from_log if bias_from_log is not None else np.zeros(3),
            scales=np.ones(3),
            gamma=np.zeros(6),
        )

        accel_ang = angle_between_vectors_deg(pose["a0"], pose["a1"])
        gyro_ang = np.degrees(quat_ang)
        err = abs(gyro_ang - accel_ang)
        scored_poses.append((err, pose))

    if len(scored_poses) < 7:
        raise RuntimeError(f"Need at least 6 valid segment pose pairs, got {len(scored_poses)}.")

    # Keep exactly 6 segments: those with the smallest gyro-vs-accel angle mismatch.
    scored_poses.sort(key=lambda t: t[0])
    poses = [pose for _, pose in scored_poses[:7]]

    # run RK4-based gyro calibration
    gyro_info = calibrate_gyro_rk4(poses, static_bias=bias_from_log, verbose=verbose)

    # compute per-segment AFTER (calibrated) diagnostics
    dt = 1.0 / SAMPLE_RATE_HZ
    per_segment_after = []
    for pose in poses:
        seg_idx = pose["segment_index"]
        a0 = pose["a0"]
        a1 = pose["a1"]
        seg = pose["gyro_samples"]
        q, R, quat_ang, vecsum_ang = integrate_gyro_segment_rk4(seg, dt=dt, bias=gyro_info["bias"],
                                                               scales=gyro_info["scale"], gamma=gyro_info["gamma"])
        a1_pred = unit_vector(R.T @ a0)
        accel_ang = angle_between_vectors_deg(a0, a1)
        pred_err = angle_between_vectors_deg(a1_pred, a1)
        per_segment_after.append({
            "segment_index": seg_idx,
            "start_line": pose["start_accel_line"],
            "end_line": pose["end_accel_line"],
            "accel_angle_deg": accel_ang,
            "gyro_quat_angle_deg": np.degrees(quat_ang),
            "pred_err_deg": pred_err,
            "n_samples": len(seg),
            "gyro_samples": seg
        })

    accel_named = {
        "bias_x": float(acc_p_arr[0]), "bias_y": float(acc_p_arr[1]), "bias_z": float(acc_p_arr[2]),
        "scale_x": float(acc_p_arr[3]), "scale_y": float(acc_p_arr[4]), "scale_z": float(acc_p_arr[5]),
        "a_yz": float(acc_p_arr[6]), "a_zy": float(acc_p_arr[7]), "a_zx": float(acc_p_arr[8]),
    }

    result = {
        "main_path": main_path,
        "accel_fit_file": accel_file_path,
        "bias_from_log": bias_from_log,
        "accel_params": accel_named,
        "accel_param_array": acc_p_arr,
        "accel_summary": {
            "mean_offset_calib": float(np.mean(np.linalg.norm(accel_fit_cal_ms2, axis=1) - g)),
            "rms_fit_calib": float(np.sqrt(np.mean((np.linalg.norm(accel_fit_cal_ms2, axis=1) - g)**2))),
        },
        "gyro_info": gyro_info,
        "per_segment_after": per_segment_after,
        "poses": poses,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result

# GUI class: supports Save dialog for naming & location and comparison panel
class CalibGUI:
    def __init__(self, master):
        self.master = master
        master.title("IMU Calibrator — Browse, Compare & Save")

        # Paths (default main path requested)
        self.main_path = tk.StringVar()
        self.main_path.set(r"C:\Users\danie\Desktop\Elu\Reaal\UT\UT_CALIB\V2 - manual_method_comprasion\DATA\gyro_clean_movements.txt")
        self.accel_path = tk.StringVar()
        self.accel_params = tk.StringVar()  # optional path or comma list

        self.last_result = None

        # Top selection frame
        frm_top = ttk.Frame(master, padding=8)
        frm_top.grid(row=0, column=0, sticky="ew")
        frm_top.columnconfigure(1, weight=1)

        ttk.Label(frm_top, text="Main log:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.main_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(frm_top, text="Browse...", command=self.browse_main).grid(row=0, column=2, padx=4)

        ttk.Label(frm_top, text="Optional accel log for params:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.accel_path).grid(row=1, column=1, sticky="ew")
        ttk.Button(frm_top, text="Browse...", command=self.browse_accel).grid(row=1, column=2, padx=4)

        ttk.Label(frm_top, text="Optional accel params file(txt or CSV):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.accel_params).grid(row=2, column=1, sticky="ew")
        ttk.Button(frm_top, text="Load file...", command=self.load_accel_params_file).grid(row=2, column=2, padx=4)

        # Buttons
        frm_buttons = ttk.Frame(master, padding=(8,4))
        frm_buttons.grid(row=1, column=0, sticky="ew")
        self.btn_run = ttk.Button(frm_buttons, text="Run calibration", command=self.run_calibration)
        self.btn_run.grid(row=0, column=0, padx=4)
        self.btn_repeat = ttk.Button(frm_buttons, text="Run again (keep paths)", command=self.run_calibration)
        self.btn_repeat.grid(row=0, column=1, padx=4)
        ttk.Button(frm_buttons, text="Export to Excel", command=self.export_excel).grid(row=0, column=2, padx=4)
        ttk.Button(frm_buttons, text="Save params (choose name/place)", command=self.save_params_dialog).grid(row=0, column=4, padx=4)
        self.btn_quit = tk.Button(
            frm_buttons,
            text="Quit",
            command=master.quit,
            bg="#c62828",
            fg="white",
            activebackground="#8e0000",
            activeforeground="white",
            relief="raised",
            padx=10,
        )
        self.btn_quit.grid(row=0, column=5, padx=4)

        # Status
        self.status = tk.StringVar(value="Idle")
        ttk.Label(master, textvariable=self.status, padding=(8,2)).grid(row=2, column=0, sticky="w")

        # Results area
        frm_res = ttk.Frame(master, padding=8)
        frm_res.grid(row=3, column=0, sticky="nsew")
        master.rowconfigure(3, weight=1)
        master.columnconfigure(0, weight=1)

        # Parameter panels
        self.accel_param_var = tk.StringVar()
        self.gyro_param_var = tk.StringVar()

        accel_panel = ttk.LabelFrame(frm_res, text="Accel Parameters", padding=8)
        accel_panel.pack(fill="x", pady=(0, 8))
        self._build_param_panel(accel_panel, self.accel_param_var, "accel")

        gyro_panel = ttk.LabelFrame(frm_res, text="Gyro Parameters", padding=8)
        gyro_panel.pack(fill="x", pady=(0, 8))
        self._build_param_panel(gyro_panel, self.gyro_param_var, "gyro")

        # Split frame: left comparison table, right summary stats table
        frame_split = ttk.Frame(frm_res)
        frame_split.pack(fill="both", expand=True)

        # Tree for per-segment accel vs gyro comparison
        left = ttk.Frame(frame_split)
        left.pack(side="left", fill="both", expand=True)
        cols = (
            "segment_index",
            "accel_angle_deg",
            "gyro_quat_angle_raw_deg",
            "pred_err_raw_deg",
            "gyro_quat_angle_cal_deg",
            "pred_err_cal_deg"
        )
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=14)
        headings = {
            "segment_index": "Segment",
            "accel_angle_deg": "Accel angle (deg)",
            "gyro_quat_angle_raw_deg": "Gyro raw (deg)",
            "pred_err_raw_deg": "Raw error (deg)",
            "gyro_quat_angle_cal_deg": "Gyro cal (deg)",
            "pred_err_cal_deg": "Cal error (deg)",
        }
        widths = {
            "segment_index": 70,
            "accel_angle_deg": 120,
            "gyro_quat_angle_raw_deg": 120,
            "pred_err_raw_deg": 110,
            "gyro_quat_angle_cal_deg": 120,
            "pred_err_cal_deg": 110,
        }
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor="center")
        self.tree.pack(fill="both", expand=True)

        # Summary stats table on the right
        right = ttk.Frame(frame_split, width=420)
        right.pack(side="right", fill="y", padx=(12, 0))
        ttk.Label(right, text="Error Summary", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0,4))
        stats_cols = ("raw_gyro", "calibrated_gyro")
        self.stats_tree = ttk.Treeview(right, columns=stats_cols, show="tree headings", height=6)
        self.stats_tree.heading("#0", text="Metric")
        self.stats_tree.column("#0", width=150, anchor="w")
        self.stats_tree.heading("raw_gyro", text="Raw gyro")
        self.stats_tree.column("raw_gyro", width=120, anchor="center")
        self.stats_tree.heading("calibrated_gyro", text="Calibrated gyro")
        self.stats_tree.column("calibrated_gyro", width=140, anchor="center")
        self.stats_tree.pack(fill="y", expand=False)

    def _build_param_panel(self, parent, param_var, prefix):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Params:").pack(side="left", padx=(0, 6))
        entry = tk.Entry(
            top,
            textvariable=param_var,
            state="readonly",
            readonlybackground="#e6e6e6",
            disabledbackground="#e6e6e6",
            relief="sunken",
            bd=1,
            font=("Courier New", 9),
            width=150,
        )
        entry.pack(side="left", fill="x", expand=True)

        matrix_row = ttk.Frame(parent)
        matrix_row.pack(fill="x")
        setattr(self, f"{prefix}_T_vars", self._create_matrix_group(matrix_row, "T", 3, 3))
        setattr(self, f"{prefix}_K_vars", self._create_matrix_group(matrix_row, "K", 3, 3))
        setattr(self, f"{prefix}_b_vars", self._create_matrix_group(matrix_row, "b", 3, 1))

    def _create_matrix_group(self, parent, label, rows, cols):
        group = ttk.Frame(parent)
        group.pack(side="left", padx=(0, 14), anchor="n")
        ttk.Label(group, text=f"{label} =").grid(row=0, column=0, sticky="w", pady=(0, 4))
        grid = ttk.Frame(group)
        grid.grid(row=1, column=0, sticky="w")

        vars_grid = []
        for row in range(rows):
            row_vars = []
            for col in range(cols):
                value_var = tk.StringVar()
                cell = tk.Entry(
                    grid,
                    textvariable=value_var,
                    state="readonly",
                    readonlybackground="#e6e6e6",
                    disabledbackground="#e6e6e6",
                    relief="sunken",
                    bd=1,
                    justify="center",
                    font=("Courier New", 8),
                    width=15,
                )
                cell.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
                row_vars.append(value_var)
            vars_grid.append(row_vars)
        return vars_grid

    def _set_matrix_vars(self, vars_grid, values):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        for row_idx, row in enumerate(vars_grid):
            for col_idx, value_var in enumerate(row):
                value_var.set(format_scientific(arr[row_idx, col_idx]))

    def _clear_matrix_vars(self, vars_grid):
        for row in vars_grid:
            for value_var in row:
                value_var.set("")

    def browse_main(self):
        p = filedialog.askopenfilename(title="Select main log file", initialdir=os.path.dirname(self.main_path.get()) or ".", filetypes=[("Text files","*.txt"),("All files","*.*")])
        if p:
            self.main_path.set(p)

    def browse_accel(self):
        p = filedialog.askopenfilename(title="Select accel-fit log file", initialdir=".", filetypes=[("Text files","*.txt"),("All files","*.*")])
        if p:
            self.accel_path.set(p)

    def load_accel_params_file(self):
        p = filedialog.askopenfilename(title="Select accel params file (.npy/.txt)", initialdir=".", filetypes=[("NumPy","*.npy"),("Text","*.txt"),("All files","*.*")])
        if p:
            self.accel_params.set(p)

    def run_calibration(self):
        main_p = self.main_path.get().strip()
        if not main_p:
            messagebox.showerror("Missing file", "Please choose a main log file first.")
            return

        self.status.set("Running calibration...")
        self._set_buttons_state("disabled")
        self.clear_results()

        thread = threading.Thread(target=self._worker_run, args=(main_p, self.accel_path.get().strip() or None, self.accel_params.get().strip() or None))
        thread.daemon = True
        thread.start()

    def _worker_run(self, main_p, accel_p, accel_params_arg):
        try:
            # parse log & compute before/after diagnostics
            res = run_pipeline(main_p, accel_file_path=accel_p, accel_params_arg=accel_params_arg, verbose=False)
            self.last_result = res

            # compute per-segment raw diagnostics as well (bias_from_log, scales=1, gamma=0)
            poses = res.get("poses", [])
            dt = 1.0 / SAMPLE_RATE_HZ
            bias_raw = res.get("bias_from_log") if res.get("bias_from_log") is not None else np.zeros(3, dtype=float)

            combined_rows = []
            stats_rows = []
            for pose in poses:
                seg = pose["gyro_samples"]
                a0 = pose["a0"]
                a1 = pose["a1"]

                # raw integration
                q_raw, R_raw, quat_ang_raw, _ = integrate_gyro_segment_rk4(seg, dt=dt, bias=bias_raw, scales=np.ones(3), gamma=np.zeros(6))
                a1_pred_raw = unit_vector(R_raw.T @ a0)
                accel_ang = angle_between_vectors_deg(a0, a1)
                pred_err_raw = angle_between_vectors_deg(a1_pred_raw, a1)

                # calibrated integration
                ginfo = res["gyro_info"]
                q_cal, R_cal, quat_ang_cal, _ = integrate_gyro_segment_rk4(seg, dt=dt, bias=ginfo["bias"], scales=ginfo["scale"], gamma=ginfo["gamma"])
                a1_pred_cal = unit_vector(R_cal.T @ a0)
                pred_err_cal = angle_between_vectors_deg(a1_pred_cal, a1)

                combined_rows.append({
                    "segment_index": pose["segment_index"],
                    "start_line": pose["start_accel_line"],
                    "end_line": pose["end_accel_line"],
                    "accel_angle_deg": accel_ang,
                    "gyro_quat_angle_raw_deg": np.degrees(quat_ang_raw),
                    "pred_err_raw_deg": pred_err_raw,
                    "gyro_quat_angle_cal_deg": np.degrees(quat_ang_cal),
                    "pred_err_cal_deg": pred_err_cal,
                    "n_samples": len(seg)
                })

            # prepare result update in main thread
            self.master.after(0, lambda: self._display_full_result(res, combined_rows))
            self.master.after(0, lambda: self.status.set(f"Done — {res['timestamp']}"))
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.master.after(0, lambda: self.status.set("Error"))
        finally:
            self.master.after(0, lambda: self._set_buttons_state("normal"))

    def _set_buttons_state(self, state):
        for child in self.master.winfo_children():
            for sub in child.winfo_children():
                try:
                    if isinstance(sub, (ttk.Button, tk.Button)):
                        sub.config(state=state)
                except Exception:
                    pass

    def clear_results(self):
        self.accel_param_var.set("")
        self.gyro_param_var.set("")
        self._clear_matrix_vars(self.accel_T_vars)
        self._clear_matrix_vars(self.accel_K_vars)
        self._clear_matrix_vars(self.accel_b_vars)
        self._clear_matrix_vars(self.gyro_T_vars)
        self._clear_matrix_vars(self.gyro_K_vars)
        self._clear_matrix_vars(self.gyro_b_vars)
        for r in self.tree.get_children():
            self.tree.delete(r)
        for r in self.stats_tree.get_children():
            self.stats_tree.delete(r)

    def _display_full_result(self, res, combined_rows):
        # fill accel & gyro texts
        ap = res["accel_params"]
        accel_bias = [ap["bias_x"], ap["bias_y"], ap["bias_z"]]
        accel_scale = [ap["scale_x"], ap["scale_y"], ap["scale_z"]]
        accel_misalignment = [ap["a_yz"], ap["a_zy"], ap["a_zx"]]
        accel_values = accel_bias + accel_scale + accel_misalignment
        accel_T = build_Ta(*accel_misalignment)
        accel_K = build_K(*accel_scale)
        accel_b = np.array(accel_bias, dtype=float)
        atxt = (
            f"{format_scientific_sequence(accel_values)}"
        )
        self.accel_param_var.set(atxt.replace("\n", " ").strip())
        self._set_matrix_vars(self.accel_T_vars, accel_T)
        self._set_matrix_vars(self.accel_K_vars, accel_K)
        self._set_matrix_vars(self.accel_b_vars, accel_b)

        g = res["gyro_info"]
        gyro_bias = [g["bias"][0], g["bias"][1], g["bias"][2]]
        gyro_scale = [g["scale"][0], g["scale"][1], g["scale"][2]]
        gyro_misalignment = [g["gamma"][0], g["gamma"][1], g["gamma"][2], g["gamma"][3], g["gamma"][4], g["gamma"][5]]
        gyro_values = gyro_bias + gyro_scale + gyro_misalignment
        gyro_T = build_Tg(*gyro_misalignment)
        gyro_K = build_K(*gyro_scale)
        gyro_b = np.array(gyro_bias, dtype=float)
        gtxt = (
            f"{format_scientific_sequence(gyro_values)}"
        )
        self.gyro_param_var.set(gtxt.replace("\n", " ").strip())
        self._set_matrix_vars(self.gyro_T_vars, gyro_T)
        self._set_matrix_vars(self.gyro_K_vars, gyro_K)
        self._set_matrix_vars(self.gyro_b_vars, gyro_b)

        # fill tree rows
        for seg in combined_rows:
            vals = (
                seg["segment_index"],
                f"{seg['accel_angle_deg']:.6f}",
                f"{seg['gyro_quat_angle_raw_deg']:.6f}",
                f"{seg['pred_err_raw_deg']:.6f}",
                f"{seg['gyro_quat_angle_cal_deg']:.6f}",
                f"{seg['pred_err_cal_deg']:.6f}"
            )
            self.tree.insert("", tk.END, values=vals)

        # compute statistics and show in comparison text
        raw_errs = np.array([float(s["pred_err_raw_deg"]) for s in combined_rows], dtype=float) if combined_rows else np.array([])
        cal_errs = np.array([float(s["pred_err_cal_deg"]) for s in combined_rows], dtype=float) if combined_rows else np.array([])
        raw_ang_diffs = np.array([abs(float(s["gyro_quat_angle_raw_deg"]) - float(s["accel_angle_deg"])) for s in combined_rows]) if combined_rows else np.array([])
        cal_ang_diffs = np.array([abs(float(s["gyro_quat_angle_cal_deg"]) - float(s["accel_angle_deg"])) for s in combined_rows]) if combined_rows else np.array([])

        def stats(arr):
            if arr.size == 0:
                return ("n/a","n/a","n/a")
            return (f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}", f"{np.max(arr):.6f}")

        raw_err_stats = stats(raw_errs)
        cal_err_stats = stats(cal_errs)
        raw_ang_stats = stats(raw_ang_diffs)
        cal_ang_stats = stats(cal_ang_diffs)

        stat_rows = [
            ("Mean error", raw_err_stats[0], cal_err_stats[0]),
            ("Median error", raw_err_stats[1], cal_err_stats[1]),
            ("Max error", raw_err_stats[2], cal_err_stats[2]),
            ("Mean |gyro-accel|", raw_ang_stats[0], cal_ang_stats[0]),
            ("Median |gyro-accel|", raw_ang_stats[1], cal_ang_stats[1]),
            ("Max |gyro-accel|", raw_ang_stats[2], cal_ang_stats[2]),
        ]
        for label, raw_value, cal_value in stat_rows:
            self.stats_tree.insert("", tk.END, text=label, values=(raw_value, cal_value))

    def export_excel(self):
        if not self.last_result:
            messagebox.showinfo("No results", "Run calibration first, then export.")
            return
        p = filedialog.asksaveasfilename(title="Save results to Excel", defaultextension=".xlsx", filetypes=[("Excel workbook","*.xlsx")])
        if not p:
            return
        try:
            res = self.last_result
            params_df = pd.DataFrame(build_sensor_parameter_rows(res))
            accel_df = pd.DataFrame([res["accel_params"]])
            gyro = res["gyro_info"]
            gyro_df = pd.DataFrame([{
                "success": gyro["success"],
                "message": gyro.get("message"),
                "cost_stage1": gyro.get("cost_stage1"),
                "cost_stage2": gyro.get("cost_stage2"),
                "bias_x": float(gyro["bias"][0]), "bias_y": float(gyro["bias"][1]), "bias_z": float(gyro["bias"][2]),
                "scale_x": float(gyro["scale"][0]), "scale_y": float(gyro["scale"][1]), "scale_z": float(gyro["scale"][2]),
                "gamma0": float(gyro["gamma"][0]), "gamma1": float(gyro["gamma"][1]), "gamma2": float(gyro["gamma"][2]),
                "gamma3": float(gyro["gamma"][3]), "gamma4": float(gyro["gamma"][4]), "gamma5": float(gyro["gamma"][5]),
            }])
            seg_df = pd.DataFrame(res["per_segment_after"])
            with pd.ExcelWriter(p, engine="openpyxl") as writer:
                params_df.to_excel(writer, sheet_name="calibration_params", index=False)
                accel_df.to_excel(writer, sheet_name="accel_params", index=False)
                gyro_df.to_excel(writer, sheet_name="gyro_params", index=False)
                seg_df.to_excel(writer, sheet_name="per_segment_after", index=False)
                workbook = writer.book
                for sheet_name in ["calibration_params", "accel_params", "gyro_params", "per_segment_after"]:
                    worksheet = workbook[sheet_name]
                    for row in worksheet.iter_rows(min_row=2):
                        for cell in row:
                            if isinstance(cell.value, (int, float)):
                                cell.number_format = "0.000000000000"
            messagebox.showinfo("Saved", f"Results saved to {p}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def save_params_dialog(self):
        "Ask user for base filename/location, then save accel & gyro params with that base."
        if not self.last_result:
            messagebox.showinfo("No results", "Run calibration first, then save params.")
            return
        # ask for base filename (user chooses path and base name; we append suffixes)
        p = filedialog.asksaveasfilename(title="Choose base filename for saving params (no extension)", defaultextension="", filetypes=[("All files","*.*")], initialfile=os.path.splitext(os.path.basename(self.main_path.get()))[0] + "_params")
        if not p:
            return
        base_dir = os.path.dirname(p) or "."
        base_name = os.path.splitext(os.path.basename(p))[0]

        try:
            res = self.last_result
            # accel
            acc_arr = res.get("accel_param_array")
            if acc_arr is None:
                ap = res["accel_params"]
                acc_arr = np.array([ap["bias_x"], ap["bias_y"], ap["bias_z"],
                                    ap["scale_x"], ap["scale_y"], ap["scale_z"],
                                    ap["a_yz"], ap["a_zy"], ap["a_zx"]], dtype=float)
            accel_txt = os.path.join(base_dir, f"{base_name}_accel_params.txt")
            np.savetxt(accel_txt, acc_arr, fmt="%.9e")

            # gyro: bias(3), scale(3), gamma(6)
            g = res["gyro_info"]
            gyro_row = np.concatenate([g["bias"], g["scale"], g["gamma"]])
            gyro_txt = os.path.join(base_dir, f"{base_name}_gyro_params.txt")
            np.savetxt(gyro_txt, gyro_row, fmt="%.9e")

            messagebox.showinfo("Saved", f"Saved accel ->\n{accel_txt}\n\nSaved gyro ->\n{gyro_txt}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

# Launch GUI
def start_gui():
    root = tk.Tk()
    root.geometry("1920x1120")
    root.minsize(1720, 980)
    app = CalibGUI(root)
    root.mainloop()

if __name__ == "__main__":
    start_gui()
# ---------------------- end GUI wrapper ----------------------