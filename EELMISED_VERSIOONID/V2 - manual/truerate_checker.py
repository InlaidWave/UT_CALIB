import re, numpy as np

LOG = "DATA/GYRO_CLEAN_ROTATIONS.txt"
STRIDE = 100  # <-- how many samples between TUS prints (set to your firmware’s value)

re_start = re.compile(r"&?MODE\s*=\s*GYRO_START", re.I)
re_end   = re.compile(r"&?MODE\s*=\s*GYRO_END", re.I)
re_tus   = re.compile(r"&?TUS\s*:?(\d+)")

segment_rates = []
all_dt = []

in_seg = False
t = []

with open(LOG) as f:
    for line in f:
        if re_start.search(line):
            in_seg = True
            t = []
            continue
        if re_end.search(line):
            if len(t) >= 2:
                dt = np.diff(np.array(t)) / 1e6  # seconds
                all_dt.extend(dt)
                # frequency of TUS lines:
                f_tus = 1.0 / np.mean(dt)
                # convert to sample rate by multiplying by stride:
                f_samples = STRIDE * f_tus
                segment_rates.append(f_samples)
            in_seg = False
            t = []
            continue
        if in_seg:
            m = re_tus.search(line)
            if m:
                t.append(int(m.group(1)))

if segment_rates:
    print("Segments analyzed:", len(segment_rates))
    print("Per-segment sample rates (Hz):", [round(x,2) for x in segment_rates])
    print("Mean sample rate: {:.3f} Hz".format(np.mean(segment_rates)))
    # Jitter based on TUS spacing inside segments:
    if all_dt:
        print("Jitter (σ): {:.4f} ms".format(np.std(all_dt) * 1e3))
else:
    print("No timestamps found inside motion segments. Check STRIDE and that TUS prints occur between GYRO_START/END.")
