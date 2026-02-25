#!/usr/bin/env python3
"""
process_runs_by_micros_improved_headers_with_interactive_plot.py

Same parsing/integration logic as your original script, with an added simple
interactive plotter at the end so you can visualize:
 - a single run by its order index, or
 - the whole dataset (all runs) in one plot.

Added features:
 - ability to flip sign of plotted angles
 - ability to plot total absolute angle (cumulative integral of |omega|) instead of axis-specific angle
 - choice of axis to plot (x, y, z) or 'abs' for total magnitude

Core parsing and integration logic is unchanged.
"""
import sys, re, math, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- config ----------
DATA_FOLDER = "SERVO_DATA"
DEFAULT_INPUT = "/mnt/data/calib_data_20260224-2156.txt"
OUT_RUN_SUMMARY = "runs_summary.csv"
OUT_SAMPLES = "runs_samples.csv"
BIAS_STRING = "&GYRO_BIAS GX2.207271GY1.248071GZ0.380602"

# regexes (robust)
re_header = re.compile(r"[&]?\s*R\s*(?P<run>\d+)\s*S\s*(?P<set>\d+)\s*V\s*(?P<v>[-\d\.eE]+)", re.IGNORECASE)
re_header_alt = re.compile(r"[&]?\s*R(?P<run>\d+)S(?P<set>\d+)V(?P<v>[-\d\.eE]+)", re.IGNORECASE)  # matches R1S1V10.00
re_t_line = re.compile(r"[&]?T(?P<ts>\d+).*?GX(?P<gx>[-\d\.eE]+)GY(?P<gy>[-\d\.eE]+)GZ(?P<gz>[-\d\.eE]+)")
re_mode_start = re.compile(r"MODE=GYRO_START", re.IGNORECASE)
re_mode_end = re.compile(r"MODE=GYRO_END", re.IGNORECASE)
re_bias = re.compile(r"GX(?P<gx>[-\d\.eE]+).*?GY(?P<gy>[-\d\.eE]+).*?GZ(?P<gz>[-\d\.eE]+)")

def parse_bias_from_string(s):
    m = re_bias.search(s)
    if not m:
        return (0.0,0.0,0.0)
    return (float(m.group("gx")), float(m.group("gy")), float(m.group("gz")))

def select_input_path():
    # prefer explicit arg, then uploaded default, then DATA_FOLDER listing
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return str(p)
        else:
            print("Provided path not found:", sys.argv[1])
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
    try:
        idx = input("Select file index (Enter for latest): ").strip()
        if idx == "":
            return str(files[-1])
        idx = int(idx)
        return str(files[idx])
    except Exception as e:
        raise FileNotFoundError("No file selected: "+str(e))

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

def main():
    path = select_input_path()
    print("Processing:", path)
    bias_from_string = parse_bias_from_string(BIAS_STRING)
    print("Using bias:", bias_from_string)

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
                # if currently collecting a run, close it first
                if current_run is not None:
                    runs.append(current_run)
                    current_run = None
                continue
            # mode start -> start collecting (use pending_hdr if any)
            if re_mode_start.search(line):
                # if already collecting a run, close and open new
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
                # if we have a pending_hdr but no current_run, start a run automatically
                if current_run is None and pending_hdr is not None:
                    current_run = {"run_idx": pending_hdr["run_idx"], "set_id": pending_hdr["set_id"],
                                   "V_cmd": pending_hdr["V_cmd"], "samples": []}
                    pending_hdr = None
                # if still no current_run, create run with unknown header (so we capture all runs)
                if current_run is None:
                    current_run = {"run_idx": None, "set_id": None, "V_cmd": float("nan"), "samples": []}
                current_run["samples"].append((ts, gx, gy, gz))
                continue
            # ignore other lines
    # end file
    # if collecting last run, append
    if current_run is not None:
        runs.append(current_run)

    print(f"Found {len(runs)} runs in file.")

    # integrate each run and assemble results
    all_samples = []
    runs_summary = []
    prev_V = None
    for order_idx, r in enumerate(runs, start=1):
        integration = integrate_run_trapezoid(r["samples"], bias_from_string)
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
    if all_samples:
        with open(OUT_SAMPLES, "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=list(all_samples[0].keys()))
            w.writeheader()
            for row in all_samples:
                w.writerow(row)
        print("Saved", OUT_SAMPLES)

    # print preview
    print("\nFirst runs summary rows:")
    for r in runs_summary[:12]:
        print(r)

    # -------------------------
    # INTERACTIVE PLOTTER (enhanced)
    # -------------------------
    # Convert to convenient structures
    if not runs_summary or not all_samples:
        print("No data to plot.")
        return

    # build lookup: order_idx -> list of samples (dicts)
    samples_by_run = {}
    # we'll also build a derived structure for each run containing:
    #  t_rel (sec), angle_x/y/z arrays, and cumulative total_abs array
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
        # compute per-sample angular rate magnitudes from gx/g y/g z/g stored in all_samples:
        gx = np.array([r["gx_u"] for r in rows_sorted], dtype=float)
        gy = np.array([r["gy_u"] for r in rows_sorted], dtype=float)
        gz = np.array([r["gz_u"] for r in rows_sorted], dtype=float)
        t_s = ts.astype(np.float64) / 1e6
        if len(t_s) >= 2:
            dt = np.diff(t_s)
            w_prev = np.sqrt(gx[:-1]**2 + gy[:-1]**2 + gz[:-1]**2)
            w_curr = np.sqrt(gx[1:]**2 + gy[1:]**2 + gz[1:]**2)
            # incremental absolute rotation for each interval
            inc = 0.5 * (w_prev + w_curr) * dt
            # cumulative total absolute angle: start at 0 and append cumulative sums matching samples
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

    # helper to plot a single run
    def plot_run(order_idx, axis="x", flip=False, abs_total=False):
        d = runs_derived.get(order_idx)
        if d is None:
            print("Run", order_idx, "not found.")
            return
        t = d["t_rel"]
        if abs_total:
            y = d["total_abs"]
            label = "total_abs (deg)"
        else:
            if axis == "x":
                y = d["angle_x"]
                label = "angle_x (deg)"
            elif axis == "y":
                y = d["angle_y"]
                label = "angle_y (deg)"
            elif axis == "z":
                y = d["angle_z"]
                label = "angle_z (deg)"
            else:
                print("Unknown axis:", axis)
                return
        if flip:
            y = -y
            label += " (flipped)"
        meta = next((rr for rr in runs_summary if rr["order_idx"] == order_idx), None)
        title = f"Run order {order_idx}"
        if meta:
            title += f" (run_idx={meta.get('run_idx')}, V_cmd={meta.get('V_cmd')})"
        plt.figure(figsize=(8,4))
        plt.plot(t, y, label=label)
        plt.xlabel("Time since run start (s)")
        plt.ylabel("Angle (deg)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # helper to plot all runs on one figure (angle_x by default, can request abs_total)
    def plot_all(axis="x", flip=False, abs_total=False):
        plt.figure(figsize=(10,6))
        for order_idx in sorted(runs_derived.keys()):
            d = runs_derived[order_idx]
            t = d["t_rel"]
            if abs_total:
                y = d["total_abs"]
            else:
                if axis == "x":
                    y = d["angle_x"]
                elif axis == "y":
                    y = d["angle_y"]
                elif axis == "z":
                    y = d["angle_z"]
                else:
                    y = d["angle_x"]
            if flip:
                y = -y
            # offsetting traces vertically would be confusing; plot all on same axes
            plt.plot(t, y, label=f"run {order_idx} (V={d.get('V_cmd')})", linewidth=0.7)
        plt.xlabel("Time since run start (s)")
        plt.ylabel("Angle (deg)")
        title = "All runs: "
        title += "total_abs" if abs_total else f"angle_{axis}"
        if flip:
            title += " (flipped)"
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # interactive loop instructions
    print("\nInteractive plotter ready. Commands:")
    print(" - Enter a run order number to plot that run (e.g. 1)")
    print(" - You can append modifiers separated by commas:")
    print("     ,flip      -> flip sign of plotted values")
    print("     ,abs       -> plot total absolute angle instead of axis")
    print("     ,axis=y    -> choose axis x/y/z (default x). Example: 3,axis=z")
    print("   Examples: '3,abs'   '5,flip'   '7,axis=y,flip' ")
    print(" - Enter 'all' (or 'all,abs') to plot all runs (angle_x by default)")
    print(" - Enter 'list' to print available runs and V_cmd")
    print(" - Press Enter (empty) to exit")

    while True:
        cmd = input("\nPlot> ").strip().lower()
        if cmd == "":
            print("Exiting plotter.")
            break
        # parse 'all' with modifiers
        parts = [p.strip() for p in cmd.split(",") if p.strip() != ""]
        if not parts:
            continue
        base = parts[0]
        modifiers = parts[1:]

        flip_flag = any(m == "flip" for m in modifiers)
        abs_flag = any(m == "abs" for m in modifiers)
        axis_flag = "x"
        for m in modifiers:
            if m.startswith("axis="):
                axis_flag = m.split("=",1)[1] or "x"
                axis_flag = axis_flag[0].lower() if axis_flag else "x"

        if base == "all":
            plot_all(axis=axis_flag, flip=flip_flag, abs_total=abs_flag)
            continue
        if base == "list":
            for rr in runs_summary:
                print(f"order {rr['order_idx']}: run_idx={rr['run_idx']} set={rr['set_id']} V_cmd={rr['V_cmd']} dur={rr['duration_s']:.3f}s")
            continue

        # try parse integer
        try:
            idx = int(base)
            plot_run(idx, axis=axis_flag, flip=flip_flag, abs_total=abs_flag)
        except ValueError:
            print("Unrecognized command. Use a run number, 'all', 'list', or Enter to quit.")
            print("Modifiers: ',flip' to invert sign, ',abs' for total absolute angle, ',axis=y' to change axis.")

if __name__ == "__main__":
    main()