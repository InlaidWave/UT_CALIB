#!/usr/bin/env python3
"""
gyrotest_parem_full.py

Parse gyro runs, integrate with trapezoid rule, save CSV summaries, and provide
an interactive plotter that uses Plotly if available, otherwise Matplotlib.
"""

import sys, re, math, csv
from pathlib import Path
import numpy as np

# ---------- config ----------
DATA_FOLDER = "SERVO_DATA"
DEFAULT_INPUT = "/mnt/data/calib_data_20260224-2156.txt"
OUT_RUN_SUMMARY = "runs_summary.csv"
OUT_SAMPLES = "runs_samples.csv"
BIAS_STRING = "&GYRO_BIAS GX2.207271GY1.248071GZ0.380602"

# regexes (robust)
re_header = re.compile(r"[&]?\s*R\s*(?P<run>\d+)\s*S\s*(?P<set>\d+)\s*V\s*(?P<v>[-\d\.eE]+)", re.IGNORECASE)
re_header_alt = re.compile(r"[&]?\s*R(?P<run>\d+)S(?P<set>\d+)V(?P<v>[-\d\.eE]+)", re.IGNORECASE)
re_t_line = re.compile(r"[&]?T(?P<ts>\d+).*?GX(?P<gx>[-\d\.eE]+)GY(?P<gy>[-\d\.eE]+)GZ(?P<gz>[-\d\.eE]+)")
re_mode_start = re.compile(r"MODE=GYRO_START", re.IGNORECASE)
re_mode_end = re.compile(r"MODE=GYRO_END", re.IGNORECASE)
re_bias = re.compile(r"GX(?P<gx>[-\d\.eE]+).*?GY(?P<gy>[-\d\.eE]+).*?GZ(?P<gz>[-\d\.eE]+)")

def build_Tg(g_yz,g_zy,g_xz,g_zx,g_xy,g_yx):
    # same convention you used elsewhere
    return np.array([[1.0, -g_yz,  g_zy],
                     [ g_xz, 1.0, -g_zx],
                     [-g_xy, g_yx, 1.0]], dtype=float)

def parse_csv_floats(s):
    # Accept "a,b,c" or "a b c" and return list of floats
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    for sep in [",", " "]:
        parts = [p.strip() for p in s.split(sep) if p.strip() != ""]
        if len(parts) >= 1:
            try:
                vals = [float(p) for p in parts]
                return vals
            except Exception:
                continue
    # last attempt: try eval-style
    try:
        vals = [float(p) for p in re.split(r"[,\s]+", s) if p]
        return vals
    except Exception:
        return None

def parse_gyro_cal_arg(s):
    """
    Parse --gyro-cal value.
    Accepts comma/space-separated floats.
    Returns tuple (bias3 or None, scale3 or None, mis6 or None).
    Interpretations:
      - 3 values  -> bias (gx,gy,gz)
      - 6 values  -> misalign (g_yz,g_zy,g_xz,g_zx,g_xy,g_yx)
      - 12 values -> bias(3), scale(3), misalign(6)
    """
    vals = parse_csv_floats(s)
    if vals is None:
        return None, None, None
    n = len(vals)
    if n == 3:
        return tuple(vals), None, None
    if n == 6:
        return None, None, tuple(vals)
    if n == 12:
        bias = tuple(vals[0:3])
        scale = tuple(vals[3:6])
        mis = tuple(vals[6:12])
        return bias, scale, mis
    raise ValueError("Invalid --gyro-cal format: expected 3, 6 or 12 numeric values (bias / misalign / bias+scale+misalign).")


def parse_bias_from_string(s):
    m = re_bias.search(s)
    if not m:
        return (0.0,0.0,0.0)
    return (float(m.group("gx")), float(m.group("gy")), float(m.group("gz")))

def select_input_path(input_path_from_cli):
    # if CLI path provided
    if input_path_from_cli:
        p = Path(input_path_from_cli)
        if p.exists():
            return str(p)
        else:
            raise FileNotFoundError(f"Provided path not found: {input_path_from_cli}")

    # try default uploaded path
    p = Path(DEFAULT_INPUT)
    if p.exists():
        return str(p)

    # fallback to DATA_FOLDER listing
    folder = Path(DATA_FOLDER)
    if not folder.exists():
        raise FileNotFoundError(f"No input and DATA_FOLDER '{DATA_FOLDER}' not found.")
    files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower()=='.txt'])
    if not files:
        raise FileNotFoundError(f"No .txt in {DATA_FOLDER}")
    print("Available files:")
    for i,f in enumerate(files):
        print(f" {i}: {f.name}")
    idx = input("Select file index (Enter for latest): ").strip()
    if idx == "":
        return str(files[-1])
    idx = int(idx)
    return str(files[idx])

def apply_gyro_calib_to_samples(samples, mis_params=None, scale_params=None, bias_params=None):
    """
    samples: list of (ts, gx, gy, gz)
    mis_params: 6-element list or None -> [g_yz,g_zy,g_xz,g_zx,g_xy,g_yx]
    scale_params: 3-element list or None -> [sx,sy,sz]
    bias_params: 3-element list or None -> [bx,by,bz]
    returns: new_samples list of (ts, gx_cal, gy_cal, gz_cal)
    """
    if not samples:
        return samples
    bias = np.array(bias_params, dtype=float) if bias_params is not None else np.zeros(3)
    scales = np.array(scale_params, dtype=float) if scale_params is not None else np.ones(3)
    if mis_params is not None:
        if len(mis_params) != 6:
            raise ValueError("misalignment requires 6 parameters (g_yz,g_zy,g_xz,g_zx,g_xy,g_yx)")
        Tg = build_Tg(*mis_params)
    else:
        Tg = np.eye(3)
    Kg = np.diag(scales)

    out = []
    for (ts, gx, gy, gz) in samples:
        v = np.array([gx, gy, gz], dtype=float)
        v_corr = Tg @ (Kg @ (v - bias))
        out.append((ts, float(v_corr[0]), float(v_corr[1]), float(v_corr[2])))
    return out

def integrate_run_trapezoid(samples, bias):
    if not samples:
        return None
    samples = sorted(samples, key=lambda s: s[0])
    ts = np.array([s[0] for s in samples], dtype=np.int64)
    gx = np.array([s[1] for s in samples], dtype=float)
    gy = np.array([s[2] for s in samples], dtype=float)
    gz = np.array([s[3] for s in samples], dtype=float)
    gx_u = gx - bias[0]; gy_u = gy - bias[1]; gz_u = gz - bias[2]
    t_s = ts.astype(np.float64) / 1e6
    if len(t_s) < 2:
        return {
            "duration_s": 0.0,
            "start_ts_us": int(ts[0]),
            "end_ts_us": int(ts[-1]),
            "N": len(ts),
            "final_ang_x_deg": 0.0,
            "final_ang_y_deg": 0.0,
            "final_ang_z_deg": 0.0,
            "total_ang_disp_deg": 0.0,
            "per_sample": [{"ts_us":int(ts[0]), "gx_u":float(gx_u[0]), "gy_u":float(gy_u[0]), "gz_u":float(gz_u[0]),
                            "angle_x_deg":0.0,"angle_y_deg":0.0,"angle_z_deg":0.0}]
        }
    dt = np.diff(t_s)
    ang_x = np.sum(0.5*(gx_u[1:]+gx_u[:-1])*dt)
    ang_y = np.sum(0.5*(gy_u[1:]+gy_u[:-1])*dt)
    ang_z = np.sum(0.5*(gz_u[1:]+gz_u[:-1])*dt)
    w_prev = np.sqrt(gx_u[:-1]**2 + gy_u[:-1]**2 + gz_u[:-1]**2)
    w_curr = np.sqrt(gx_u[1:]**2 + gy_u[1:]**2 + gz_u[1:]**2)
    total_disp = np.sum(0.5*(w_prev + w_curr)*dt)
    per_sample = []
    cumx = cumy = cumz = 0.0
    per_sample.append({"ts_us": int(ts[0]), "gx_u": float(gx_u[0]), "gy_u": float(gy_u[0]), "gz_u": float(gz_u[0]),
                       "angle_x_deg": cumx, "angle_y_deg": cumy, "angle_z_deg": cumz})
    for i in range(1, len(ts)):
        dti = float(t_s[i] - t_s[i-1])
        step_x = 0.5*(gx_u[i]+gx_u[i-1])*dti
        step_y = 0.5*(gy_u[i]+gy_u[i-1])*dti
        step_z = 0.5*(gz_u[i]+gz_u[i-1])*dti
        cumx += step_x; cumy += step_y; cumz += step_z
        per_sample.append({"ts_us": int(ts[i]), "gx_u": float(gx_u[i]), "gy_u": float(gy_u[i]), "gz_u": float(gz_u[i]),
                           "angle_x_deg": float(cumx), "angle_y_deg": float(cumy), "angle_z_deg": float(cumz)})
    return {
        "duration_s": float(t_s[-1] - t_s[0]),
        "start_ts_us": int(ts[0]),
        "end_ts_us": int(ts[-1]),
        "N": int(len(ts)),
        "final_ang_x_deg": float(ang_x),
        "final_ang_y_deg": float(ang_y),
        "final_ang_z_deg": float(ang_z),
        "total_ang_disp_deg": float(total_disp),
        "per_sample": per_sample
    }

def interactive_plotter(all_samples, runs_summary):
    """
    Interactive plotting helper. Uses Plotly if available, otherwise Matplotlib.
    Call this from main() after all_samples and runs_summary are prepared.
    """
    # safe availability checks (no importlib.util)
    try:
        import plotly.graph_objects as go
        _have_plotly = True
    except Exception:
        go = None
        _have_plotly = False

    try:
        import mplcursors
        _have_mplcursors = True
    except Exception:
        _have_mplcursors = False

    try:
        import seaborn as sns
        sns.set_theme(context="notebook", style="darkgrid")
    except Exception:
        pass

    import matplotlib.pyplot as plt

    # build lookup
    samples_by_run = {}
    runs_derived = {}
    for s in all_samples:
        samples_by_run.setdefault(s["order_idx"], []).append(s)

    for order_idx, rows in samples_by_run.items():
        rows_sorted = sorted(rows, key=lambda r: r["ts_us"])
        ts = np.array([r["ts_us"] for r in rows_sorted], dtype=np.int64)
        t_rel = (ts - ts[0]) / 1e6
        angle_x = np.array([r["angle_x_deg"] for r in rows_sorted], dtype=float)
        angle_y = np.array([r["angle_y_deg"] for r in rows_sorted], dtype=float)
        angle_z = np.array([r["angle_z_deg"] for r in rows_sorted], dtype=float)
        gx = np.array([r["gx_u"] for r in rows_sorted], dtype=float)
        gy = np.array([r["gy_u"] for r in rows_sorted], dtype=float)
        gz = np.array([r["gz_u"] for r in rows_sorted], dtype=float)
        t_s = ts.astype(np.float64) / 1e6
        if len(t_s) >= 2:
            dt = np.diff(t_s)
            w_prev = np.sqrt(gx[:-1]**2 + gy[:-1]**2 + gz[:-1]**2)
            w_curr = np.sqrt(gx[1:]**2 + gy[1:]**2 + gz[1:]**2)
            inc = 0.5 * (w_prev + w_curr) * dt
            cum_abs = np.concatenate(([0.0], np.cumsum(inc)))
        else:
            cum_abs = np.array([0.0])

        runs_derived[order_idx] = {
            "t_rel": t_rel,
            "angle_x": angle_x,
            "angle_y": angle_y,
            "angle_z": angle_z,
            "total_abs": cum_abs,
            "V_cmd": rows_sorted[0].get("V_cmd", float("nan")) if rows_sorted else float("nan")
        }

    # plotting helpers
    def _plotly_single(order_idx, axis="x", flip=False, abs_total=False, save_html=None):
        d = runs_derived.get(order_idx)
        if d is None:
            print("Run", order_idx, "not found.")
            return
        t = d["t_rel"]
        y = d["total_abs"] if abs_total else {"x": d["angle_x"], "y": d["angle_y"], "z": d["angle_z"]}.get(axis, d["angle_x"])
        if flip: y = -y
        meta = next((rr for rr in runs_summary if rr["order_idx"] == order_idx), None)
        title = f"Run order {order_idx}"
        if meta:
            title += f" (run_idx={meta.get('run_idx')}, V_cmd={meta.get('V_cmd')})"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers" if len(t) < 300 else "lines", name="angle"))
        fig.update_layout(title=title, xaxis_title="Time since run start (s)", yaxis_title="Angle (deg)", template="plotly_white")
        if save_html:
            fig.write_html(save_html, include_plotlyjs="cdn")
            print("Saved interactive HTML to", save_html)
        fig.show()

    def _plotly_all(axis="x", flip=False, abs_total=False, save_html=None):
        fig = go.Figure()
        for order_idx in sorted(runs_derived.keys()):
            d = runs_derived[order_idx]
            t = d["t_rel"]
            y = d["total_abs"] if abs_total else {"x": d["angle_x"], "y": d["angle_y"], "z": d["angle_z"]}.get(axis, d["angle_x"])
            if flip: y = -y
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"run {order_idx} (V={d.get('V_cmd')})", line=dict(width=1)))
        title = "All runs: " + ("total_abs" if abs_total else f"angle_{axis}")
        if flip: title += " (flipped)"
        fig.update_layout(title=title, xaxis_title="Time since run start (s)", yaxis_title="Angle (deg)", template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        if save_html:
            fig.write_html(save_html, include_plotlyjs="cdn")
            print("Saved interactive HTML to", save_html)
        fig.show()

    def _mpl_single(order_idx, axis="x", flip=False, abs_total=False):
        d = runs_derived.get(order_idx)
        if d is None:
            print("Run", order_idx, "not found.")
            return
        t = d["t_rel"]
        y = d["total_abs"] if abs_total else {"x": d["angle_x"], "y": d["angle_y"], "z": d["angle_z"]}.get(axis, d["angle_x"])
        if flip: y = -y
        meta = next((rr for rr in runs_summary if rr["order_idx"] == order_idx), None)
        title = f"Run order {order_idx}"
        if meta:
            title += f" (run_idx={meta.get('run_idx')}, V_cmd={meta.get('V_cmd')})"
        plt.rcParams.update({"figure.dpi": 140, "lines.linewidth": 1.4, "font.size": 11})
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(t, y, marker='o' if len(t) < 300 else None, markersize=4)
        ax.set_xlabel("Time since run start (s)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        if _have_mplcursors:
            try:
                import mplcursors
                mplcursors.cursor(ax.get_lines(), hover=True)
            except Exception:
                pass
        plt.show()

    def _mpl_all(axis="x", flip=False, abs_total=False):
        plt.rcParams.update({"figure.dpi": 120, "lines.linewidth": 1.0, "font.size": 10})
        fig, ax = plt.subplots(figsize=(10,6))
        for order_idx in sorted(runs_derived.keys()):
            d = runs_derived[order_idx]
            t = d["t_rel"]
            y = d["total_abs"] if abs_total else {"x": d["angle_x"], "y": d["angle_y"], "z": d["angle_z"]}.get(axis, d["angle_x"])
            if flip: y = -y
            ax.plot(t, y, label=f"run {order_idx} (V={d.get('V_cmd')})", linewidth=0.8)
        title = "All runs: " + ("total_abs" if abs_total else f"angle_{axis}")
        if flip: title += " (flipped)"
        ax.set_xlabel("Time since run start (s)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(title)
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)
        plt.tight_layout()
        if _have_mplcursors:
            try:
                import mplcursors
                mplcursors.cursor(ax.get_lines(), hover=True)
            except Exception:
                pass
        plt.show()

    # interactive loop
    print("\nInteractive plotter ready. Commands:")
    print(" - Enter a run order number to plot that run (e.g. 1)")
    print(" - Modifiers (comma separated):")
    print("     flip      -> flip sign of plotted values")
    print("     abs       -> plot total absolute angle instead of axis")
    print("     axis=y    -> choose axis x/y/z (default x)")
    print("     html=PATH -> (plotly only) save interactive HTML to PATH and open it")
    print("   Examples: '3,abs'   '5,flip'   '7,axis=y,flip'  '2,html=run2.html'")
    print(" - Enter 'all' (or 'all,abs') to plot all runs")
    print(" - Enter 'list' to print available runs and V_cmd")
    print(" - Press Enter (empty) to exit")

    while True:
        cmd = input("\nPlot> ").strip()
        if cmd == "":
            print("Exiting plotter.")
            break
        parts = [p.strip() for p in cmd.split(",") if p.strip() != ""]
        if not parts:
            continue
        base = parts[0].lower()
        modifiers = parts[1:]
        flip_flag = any(m.lower() == "flip" for m in modifiers)
        abs_flag = any(m.lower() == "abs" for m in modifiers)
        axis_flag = "x"
        save_html = None
        for m in modifiers:
            ml = m.lower()
            if ml.startswith("axis="):
                axis_flag = ml.split("=",1)[1] or "x"
                axis_flag = axis_flag[0].lower()
            if ml.startswith("html="):
                save_html = m.split("=",1)[1]

        if base == "all":
            if _have_plotly and go is not None:
                _plotly_all(axis=axis_flag, flip=flip_flag, abs_total=abs_flag, save_html=save_html)
            else:
                _mpl_all(axis=axis_flag, flip=flip_flag, abs_total=abs_flag)
            continue
        if base == "list":
            for rr in runs_summary:
                print(f"order {rr['order_idx']}: run_idx={rr['run_idx']} set={rr['set_id']} V_cmd={rr['V_cmd']} dur={rr['duration_s']:.3f}s")
            continue
        try:
            idx = int(base)
            if _have_plotly and go is not None:
                _plotly_single(idx, axis=axis_flag, flip=flip_flag, abs_total=abs_flag, save_html=save_html)
            else:
                _mpl_single(idx, axis=axis_flag, flip=flip_flag, abs_total=abs_flag)
        except ValueError:
            print("Unrecognized command. Use a run number, 'all', 'list', or Enter to quit.")
            print("Modifiers: ',flip' to invert sign, ',abs' for total absolute angle, ',axis=y' to change axis, 'html=PATH' to save plotly HTML.")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Process gyro runs and optionally apply gyro calibration.")
    ap.add_argument("input_path", nargs="?", default=None,
                    help="Path to input .txt file (optional). If omitted, the script will try DEFAULT_INPUT or list DATA_FOLDER.")
    ap.add_argument("--gyro-cal", type=str, default=None,
                    help='Single compact gyro calibration: 3=bias, 6=misalign, 12=bias(3),scale(3),misalign(6). Comma or space separated.')
    ap.add_argument("--gyro-bias", type=str, default=None,
                    help='Comma or space-separated gyro bias: "gx,gy,gz" (will override file/BIAS_STRING if provided).')
    ap.add_argument("--gyro-scale", type=str, default=None,
                    help='Comma-separated scales: "sx,sy,sz" (applied after bias subtraction).')
    ap.add_argument("--gyro-misalign", type=str, default=None,
                    help='Six misalignment params: "g_yz,g_zy,g_xz,g_zx,g_xy,g_yx" (applied as Tg).')
    args = ap.parse_args()

    # choose path (prefer explicit arg, then selector)
    if args.input_path:
        path = args.input_path
    else:
        try:
            path = select_input_path(args.input_path)
        except Exception as e:
            sys.exit(f"Failed to select input file: {e}")

    # parse CLI gyro cal args (prefer --gyro-cal single arg)
    cli_bias = cli_scale = cli_mis = None
    if args.gyro_cal:
        try:
            cli_bias, cli_scale, cli_mis = parse_gyro_cal_arg(args.gyro_cal)
        except Exception as e:
            sys.exit(f"Failed to parse --gyro-cal: {e}")
    else:
        cli_bias = parse_csv_floats(args.gyro_bias)
        cli_scale = parse_csv_floats(args.gyro_scale)
        cli_mis   = parse_csv_floats(args.gyro_misalign)

    # validate sizes if present
    if cli_bias is not None and len(cli_bias) != 3:
        sys.exit("Expected 3 values for gyro bias.")
    if cli_scale is not None and len(cli_scale) != 3:
        sys.exit("Expected 3 values for gyro scale.")
    if cli_mis is not None and len(cli_mis) != 6:
        sys.exit("Expected 6 values for gyro misalignment.")

    print("Processing:", path)
    bias_from_string = parse_bias_from_string(BIAS_STRING)
    print("Default bias from BIAS_STRING:", bias_from_string)

    # parse file into runs (same as before)
    runs = []
    pending_hdr = None
    current_run = None

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            line = ln.strip()
            if not line:
                continue
            # file-level bias override
            if "&GYRO_BIAS" in line:
                b = parse_bias_from_string(line)
                print("Bias override found in file:", b)
                bias_from_string = b
                continue
            # try header patterns (two variants)
            mh = re_header.search(line) or re_header_alt.search(line)
            if mh:
                hdr = {"run_idx": int(mh.group("run")), "set_id": int(mh.group("set")), "V_cmd": float(mh.group("v")), "hdr_line": line}
                pending_hdr = hdr
                if current_run is not None:
                    runs.append(current_run)
                    current_run = None
                continue
            # mode start -> start collecting (use pending_hdr if any)
            if re_mode_start.search(line):
                if current_run is not None:
                    runs.append(current_run)
                current_run = {"run_idx": pending_hdr["run_idx"] if pending_hdr else None,
                               "set_id": pending_hdr["set_id"] if pending_hdr else None,
                               "V_cmd": pending_hdr["V_cmd"] if pending_hdr else float("nan"),
                               "samples": []}
                pending_hdr = None
                continue
            # mode end -> close run
            if re_mode_end.search(line):
                if current_run is not None:
                    runs.append(current_run)
                    current_run = None
                pending_hdr = None
                continue
            # data line
            m = re_t_line.search(line)
            if m:
                ts = int(m.group("ts")); gx = float(m.group("gx")); gy = float(m.group("gy")); gz = float(m.group("gz"))
                if current_run is None and pending_hdr is not None:
                    current_run = {"run_idx": pending_hdr["run_idx"], "set_id": pending_hdr["set_id"],
                                   "V_cmd": pending_hdr["V_cmd"], "samples": []}
                    pending_hdr = None
                if current_run is None:
                    current_run = {"run_idx": None, "set_id": None, "V_cmd": float("nan"), "samples": []}
                current_run["samples"].append((ts, gx, gy, gz))
                continue
            # ignore other lines
    if current_run is not None:
        runs.append(current_run)

    print(f"Found {len(runs)} runs in file.")

    # If CLI calibration provided, prepare values and apply to each run's samples.
    use_cli_cal = (cli_bias is not None) or (cli_scale is not None) or (cli_mis is not None)
    if use_cli_cal:
        print("CLI gyro calibration provided. Applying to samples before integration.")
        print(" CLI bias:", cli_bias)
        print(" CLI scale:", cli_scale)
        print(" CLI misalign:", cli_mis)
        # Apply to each run's samples
        for r in runs:
            r["samples"] = apply_gyro_calib_to_samples(r["samples"],
                                                      mis_params=cli_mis,
                                                      scale_params=cli_scale,
                                                      bias_params=cli_bias)
        # after applying calibration, integration should not subtract additional bias:
        bias_for_integration = (0.0, 0.0, 0.0)
    else:
        # use parsed string bias (BIAS_STRING or file override) as integrator bias
        bias_for_integration = bias_from_string

    # integrate each run and assemble results
    all_samples = []
    runs_summary = []
    prev_V = None
    for order_idx, r in enumerate(runs, start=1):
        integration = integrate_run_trapezoid(r["samples"], bias_for_integration)
        if integration is None:
            continue
        duration = integration["duration_s"]
        curr_V = r.get("V_cmd", float("nan"))
        avg_speed_used = float("nan")
        if prev_V is not None and not math.isnan(curr_V):
            avg_speed_used = 0.5 * (prev_V + curr_V)
        commanded_angle = avg_speed_used * duration if not math.isnan(avg_speed_used) else float("nan")

        runs_summary.append({
            "order_idx": order_idx,
            "run_idx": r.get("run_idx"),
            "set_id": r.get("set_id"),
            "start_ts_us": integration["start_ts_us"],
            "end_ts_us": integration["end_ts_us"],
            "duration_s": integration["duration_s"],
            "V_cmd": curr_V,
            "avg_speed_used": avg_speed_used,
            "commanded_angle_est_deg": commanded_angle,
            "final_ang_x_deg": integration["final_ang_x_deg"],
            "final_ang_y_deg": integration["final_ang_y_deg"],
            "final_ang_z_deg": integration["final_ang_z_deg"],
            "total_ang_disp_deg": integration["total_ang_disp_deg"],
            "N_samples": integration["N"]
        })

        for s in integration["per_sample"]:
            all_samples.append({
                "order_idx": order_idx,
                "run_idx": r.get("run_idx"),
                "set_id": r.get("set_id"),
                "V_cmd": curr_V,
                "ts_us": s["ts_us"],
                "gx_u": s["gx_u"],
                "gy_u": s["gy_u"],
                "gz_u": s["gz_u"],
                "angle_x_deg": s["angle_x_deg"],
                "angle_y_deg": s["angle_y_deg"],
                "angle_z_deg": s["angle_z_deg"]
            })

        prev_V = curr_V

    # save CSVs
    if runs_summary:
        with open(OUT_RUN_SUMMARY, "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=list(runs_summary[0].keys()))
            w.writeheader()
            for row in runs_summary:
                w.writerow(row)
        print("Saved", OUT_RUN_SUMMARY)

    # preview
    print("\nFirst runs summary rows:")
    for r in runs_summary[:12]:
        print(r)

    # call interactive plotter
    if runs_summary and all_samples:
        interactive_plotter(all_samples, runs_summary)
    else:
        print("No data to plot.")

if __name__ == "__main__":
    main()