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
def list_txt_files(directory):
    folder_path = os.path.join(directory, data_folder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# =========================
# Parse log
# =========================
accel_pat = re.compile(r"&?X\s*([-+]?\d*\.?\d+)\s*Y\s*([-+]?\d*\.?\d+)\s*Z\s*([-+]?\d*\.?\d+)")
gyro_pat  = re.compile(r"&?GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)")
mode_start = re.compile(r"MODE\s*=\s*GYRO_START", re.I)
mode_end   = re.compile(r"MODE\s*=\s*GYRO_END", re.I)

def parse_log(path:str)->Tuple[np.ndarray,List[np.ndarray], np.ndarray]:
    accel, gyros, seg, in_seg = [], [], [], False

    bias = None
    gyro_bias_pat = re.compile(
        r"&?GYRO_BIAS.*?GX\s*([-+]?\d*\.?\d+)\s*GY\s*([-+]?\d*\.?\d+)\s*GZ\s*([-+]?\d*\.?\d+)",
        re.I
    )

    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            m_bias = gyro_bias_pat.search(line)
            if m_bias:
                bias = np.array(list(map(float, m_bias.groups())))
                continue
            if mode_start.search(line):
                if seg: gyros.append(np.array(seg)); seg=[]
                in_seg=True; continue
            if mode_end.search(line):
                if seg: gyros.append(np.array(seg)); seg=[]
                in_seg=False; continue
            m=gyro_pat.search(line)
            if m and in_seg:
                seg.append(list(map(float,m.groups())))
                continue
            m=accel_pat.search(line)
            if m:
                accel.append(list(map(float,m.groups())))
    if seg: gyros.append(np.array(seg))
    return np.array(accel,float), gyros, bias

# =========================
# Accelerometer calibration
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
# Gyro helpers (integrators)
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
# Gyro calibration (residuals)
# =========================
def build_gyro_residuals(accel_dirs, gyro_segments, params):
    idx=0
    if GYRO_FIT_MISALIGNMENT:
        g_yz,g_zy,g_xz,g_zx,g_xy,g_yx=params[idx:idx+6]; idx+=6
        Tg=build_Tg(g_yz,g_zy,g_xz,g_zx,g_xy,g_yx)
    else:
        Tg=np.eye(3)
    sgx,sgy,sgz=params[idx:idx+3]; idx+=3
    Kg=np.diag([sgx,sgy,sgz])
    bg=np.array(params[idx:idx+3]) if idx<len(params) else np.zeros(3)
    dt=1.0/SAMPLE_RATE_HZ
    res=[]
    for k in range(1,len(accel_dirs)):
        a0,a1=accel_dirs[k-1],accel_dirs[k]
        seg=gyro_segments[k-1]
        if seg.size==0: continue
        omega=(Tg@(Kg@(seg.T+bg[:,None]))).T
        omega_r=omega_to_rad_s(omega)

        if GYRO_INTEGRATOR == "rk4":
            q_final = rk4n_integrate_quat(omega_r, dt)
            R = quat_to_R(q_final)
            a_pred = R @ a0
        elif GYRO_INTEGRATOR == "rodrigues":
            dtheta=np.sum(omega_r,axis=0)*dt
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
        else:  # 'smallangle'
            dtheta=np.sum(omega_r,axis=0)*dt
            a_pred = a0 - np.cross(dtheta, a0)

        res.append(a_pred-a1)
    return np.concatenate(res) if res else np.zeros(0)

def calibrate_gyro(accel_calib, gyro_segments, static_bias=None):
    dirs = accel_calib / np.linalg.norm(accel_calib, axis=1, keepdims=True)
    p = []

    if GYRO_FIT_MISALIGNMENT:
        p += [0] * 6
    p += [1, 1, 1]

    use_static_bias = static_bias is not None
    if not use_static_bias and GYRO_CALIB_MODE == "fit_bias":
        p += [0, 0, 0]
    p = np.array(p, float)

    def fun(x):
        x_full = np.concatenate([x, static_bias]) if use_static_bias else x
        return build_gyro_residuals(dirs, gyro_segments, x_full)

    print(f"Starting gyro optimization (<5 s with integrator='{GYRO_INTEGRATOR}')...")
    res = least_squares(fun, p,
                    max_nfev=300,
                    verbose=2,
                    loss='soft_l1',
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6)

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
# Accel points visualizer
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

def compare_cross_calibration(target_path: str, calib_params: np.ndarray):
    print("\n=== Cross-dataset calibration test ===")
    print(f"Using parameters from reference run on: {os.path.basename(target_path)}")
    accel_new, _, _ = parse_log(target_path)
    if accel_new.size == 0:
        print("No accelerometer data found in target file.")
        return
    mean_norm = np.mean(np.linalg.norm(accel_new, axis=1))
    accel_new_ms2 = accel_new * g if 0.3 <= mean_norm <= 2 else accel_new
    accel_calib_ms2 = accel_apply(accel_new_ms2, calib_params)
    accel_summary(accel_calib_ms2, accel_new_ms2)

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

    if args.accel_file:
        accel, _, _ = parse_log(args.accel_file)
        _, gyros, bias = parse_log(path)
    else:
        accel, gyros, bias = parse_log(path)

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
                # inline comma-separated list
                values = [float(v.strip()) for v in args.accel_params.split(",")]
                if len(values) != 9:
                    raise ValueError("Expected 9 parameters (bx,by,bz,sx,sy,sz,a_yz,a_zy,a_zx)")
                acc_p = np.array(values, dtype=float)
                print("\nLoaded accelerometer parameters from command line:")
            print(" One row:", " ".join(f"{x:.8e}" for x in acc_p))
        except Exception as e:
            print(f"Failed to parse --accel-params: {e}")
            sys.exit(1)

    # Accelerometer calibration / application
    if accel.size == 0:
        sys.exit("No accel data found.")
    mean_norm = np.mean(np.linalg.norm(accel, axis=1))
    accel_raw = accel * g if 0.3 <= mean_norm <= 2 else accel

    if acc_p is None:
        acc_p = accel_find_params(accel_raw)
        print("\nComputed accelerometer calibration:")
        print(" Bias:", acc_p[:3])
        print(" Scales:", acc_p[3:6])
        print(" Misalignment (a_yz, a_zy, a_zx):", acc_p[6:])
        print(" One row:", " ".join(f"{x:.8e}" for x in acc_p))
        accel_cal = accel_apply(accel_raw, acc_p)
        accel_summary(accel_cal, accel_raw)
    else:
        print("\nUsing provided accelerometer calibration parameters.")
        accel_cal = accel_apply(accel_raw, acc_p)

    compare_cross_calibration("DATA/big_turntable_test_sample.txt", acc_p)

# =========================
# Gyroscope calibration
# =========================
if gyros:
    print(f"\nFound {len(gyros)} gyro segments.")
    if len(gyros) != len(accel_cal) - 1:
        n = min(len(gyros), len(accel_cal) - 1)
        gyros = gyros[:n]
        accel_cal = accel_cal[:n + 1]

    # ---- FILTER OUT STATIC OR NEAR-STATIC SEGMENTS ----
    print("Filtering gyro segments for real motion...")

    # # 1. Estimate bias
    static_bias = bias if bias is not None and bias.size == 3 else None
    if static_bias is not None:
        bias_est = static_bias
    else:
        all_samples = np.vstack([seg for seg in gyros if seg.size > 0])
        bias_est = np.mean(all_samples, axis=0)
    print(f"Estimated gyro bias for filtering: {bias_est}")

    # # 2. Filter out individual low-motion readings inside each segment
    # min_rate_threshold = 0.3  # deg/s
    # valid_segments = []
    # kept_readings = 0
    # discarded_readings = 0

    # for seg_idx, seg in enumerate(gyros):
    #     if seg.size == 0:
    #         continue
    #     seg_detrended = seg - bias_est
    #     magnitudes = np.linalg.norm(seg_detrended, axis=1)

    #     valid_mask = magnitudes > min_rate_threshold
    #     valid_count = np.sum(valid_mask)

    #     print(f"Segment {seg_idx}: {valid_count}/{len(seg)} samples above {min_rate_threshold} deg/s")

    #     if valid_count < 5:
    #         discarded_readings += seg.shape[0]
    #         print(f"  Discarded segment {seg_idx} (too static)")
    #         continue

    #     seg_filtered = seg[valid_mask]
    #     valid_segments.append(seg_filtered)
    #     kept_readings += seg_filtered.shape[0]
    #     discarded_readings += np.sum(~valid_mask)

    # gyros = valid_segments
    # print(f"Kept {len(gyros)} segments ({kept_readings} samples kept, {discarded_readings} discarded)\n")

    # 3. Calibrate
    _, gyro_info = calibrate_gyro(accel_cal, gyros, static_bias)

    print("\nGyro calibration results:")
    if "gamma" in gyro_info:
        print(" Misalign gamma:", gyro_info["gamma"])
    print(" Scale:", gyro_info["scale"])
    print(" Bias:", gyro_info["bias"])
    print(" Success:", gyro_info["success"], " Cost:", gyro_info["cost"])
else:
    print("No gyro segments found.")