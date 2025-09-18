import os
import numpy as np
from scipy.optimize import least_squares

data_folder = "DATA"
data_log = []
g = 9.81872  # local gravity magnitude

def list_txt_files(directory):
    folder_path = os.path.join(directory, data_folder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

def read_file_data(file_path):
    global data_log
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("MODE="):
                continue
            else:
                line = line.strip()
                x_start = line.find('X') + 1
                y_start = line.find('Y') + 1
                z_start = line.find('Z') + 1

                x_value = line[x_start:line.find('Y')]
                y_value = line[y_start:line.find('Z')]
                z_value = line[z_start:]

                data_log.append([float(x_value), float(y_value), float(z_value)])

    data_log = np.array(data_log) * g  # convert to numpy array and units of m/s^2 for further calculations
    print(f"Loaded {len(data_log)} samples.")

def build_T(beta_yz, beta_zy, beta_xz, beta_zx, beta_xy, beta_yx):
    """Construct the non-orthogonality matrix T^a"""
    return np.array([
        [1.0,     -beta_yz,  beta_zy],
        [beta_xz,  1.0,     -beta_zx],
        [-beta_xy, beta_yx,  1.0]
    ])

def build_K(sx, sy, sz):
    """Construct the diagonal scale matrix K^a"""
    return np.diag([sx, sy, sz])

def residuals(params, data):
    """
    params = [bx, by, bz, sx, sy, sz, beta_yz, beta_zy, beta_xz, beta_zx, beta_xy, beta_yx]
    """
    bx, by, bz, sx, sy, sz, beta_yz, beta_zy, beta_xz, beta_zx, beta_xy, beta_yx = params

    b = np.array([bx, by, bz])
    K = build_K(sx, sy, sz)
    T = build_T(beta_yz, beta_zy, beta_xz, beta_zx, beta_xy, beta_yx)

    res = []
    for a in data:
        a_corr = T @ (K @ (a + b))
        res.append(np.linalg.norm(a_corr) - g)
    return res

def find_calib_parameters(meas):
    # Initial guess that should be close: 0 bias, near-identity scaling, no misalignments
    init_params = [0,0,0, 1,1,1, 0,0,0,0,0,0]
    result = least_squares(residuals, init_params, args=(data_log,)) # multiplication by g to transform into units of m/s^2
    return result.x

def calibrate(data_log, params):
    # Extract parameters from params
    bx, by, bz = params[:3]     # biases
    sx, sy, sz = params[3:6]    # scale factors
    beta_yz, beta_zy, beta_xz, beta_zx, beta_xy, beta_yx = params[6:]   # misalignments

    b = np.array([bx, by, bz])
    K = np.diag([sx, sy, sz])
    T = np.array([
    [1.0,     -beta_yz,  beta_zy],
    [beta_xz,  1.0,     -beta_zx],
    [-beta_xy, beta_yx,  1.0]
    ])

    return (T @ (K @ (data_log.T + b[:, np.newaxis]))).T  # shape (N,3)     (b is made a column vector(3;1))

def percentile_mean_error(residuals, percentile=95):
    cutoff = np.percentile(np.abs(residuals), percentile)
    top_errors = np.abs(residuals)[np.abs(residuals) >= cutoff]
    return np.mean(top_errors)

def compare(data_log_1, data_log_2):
    magnitudes_1 = np.linalg.norm(data_log_1, axis=1)
    magnitudes_2 = np.linalg.norm(data_log_2, axis=1)

    # Compute residuals relative to g
    residuals_from_g_1 = magnitudes_1 - g
    residuals_from_g_2 = magnitudes_2 - g

    # Summary statistics
    mean_error_1 = np.mean(residuals_from_g_1)
    mean_error_2 = np.mean(residuals_from_g_2)

    rms_error_1 = np.sqrt(np.mean(residuals_from_g_1**2))
    rms_error_2 = np.sqrt(np.mean(residuals_from_g_2**2))
    
    # max_error_1 = np.max(np.abs(residuals_from_g_1))
    # max_error_2 = np.max(np.abs(residuals_from_g_2))

    five_percentile_error_1 = percentile_mean_error(residuals_from_g_1, 95)
    five_percentile_error_2 = percentile_mean_error(residuals_from_g_2, 95)

    print("Mean offset from g for calibrated data:", mean_error_1)
    print("Mean offset from g for uncalibrated data:", mean_error_2)

    print("5 percent offset from g for calibrated data:", five_percentile_error_1)
    print("5 percent offset from g for uncalibrated data:", five_percentile_error_2)

    print("fit to gravity sphere for calibrated data:", rms_error_1)
    print("fit to gravity sphere for uncalibrated data:", rms_error_2)



if __name__ == "__main__":
    txt_files = list_txt_files(".")
    print("Available .txt files:")
    for idx, fname in enumerate(txt_files):
        print(f"{idx}: {fname}")
    choice = int(input("Select a file by number: "))
    data_file = os.path.join(data_folder, txt_files[choice])
    read_file_data(data_file)

    # Perform calibration
    params = find_calib_parameters(data_log)
    calibrated_data_log = calibrate(data_log, params)
    compare(calibrated_data_log, data_log)
    # print("\nCalibration parameters found:")
    # print(f"Bias (bx,by,bz): {params[:3]}")
    # print(f"Scales (sx,sy,sz): {params[3:6]}")
    # print(f"Non-orthogonality betas (yz, zy, xz, zx, xy, yx): {params[6:]}")
