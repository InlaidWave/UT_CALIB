import os
import math
import numpy as np

mode = None
data_folder = "DATA"
data_log = []

def list_txt_files(directory):
    folder_path = os.path.join(directory, data_folder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

def read_file_data(file_path):
    data_log_position = -1
    print(file_path)

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("MODE="):
                mode_f = line[len("MODE="):]  # Gets everything after 'MODE='
                if mode_f == "M_ACCEL":
                    mode = "Manual testbench accelerometer calibration"
                elif mode_f == "S_ACCEL":
                    mode = "Motorized testbench accelerometer calibration"
                else:
                    mode = mode_f   # makes mode raw mode name if not recognised
                print("Calibration mode:", mode)
            else:
                data_log_position += 1
                line = line.strip()
                x_start = line.find('X') + 1
                y_start = line.find('Y') + 1
                z_start = line.find('Z') + 1
                axis_start = line.find('A') + 1

                x_value = line[x_start:line.find('Y')]
                y_value = line[y_start:line.find('Z')]
                z_value = line[z_start:line.find('A')]
                axis = line[axis_start:]

                pos_deg = (data_log_position + 1) * 15

                data_log.append({
                    "x": x_value,
                    "y": y_value,
                    "z": z_value,
                    "axis": axis,
                    "position on axis(deg)": pos_deg
                })

                print(data_log[data_log_position])
    
    print("-----------------")
    remove_tilt()

def remove_tilt():
    averages = {}

    step = 15
    max_deg = 360

    for deg in range(step, max_deg // 2, step):
        pair = (deg, max_deg - deg)
        readings_pair = [r for r in data_log if r["position on axis(deg)"] in pair] # make a list including only two opposite positions on an axis
        if readings_pair and 180 not in pair:
            x_avg = sum(float(r["x"]) for r in readings_pair) / len(readings_pair)
            y_avg = sum(float(r["y"]) for r in readings_pair) / len(readings_pair)
            z_avg = sum(float(r["z"]) for r in readings_pair) / len(readings_pair)
            averages[pair] = {"x": x_avg, "y": y_avg, "z": z_avg}
        
    special_pair = (180, 360)
    readings_special = [r for r in data_log if r["position on axis(deg)"] in special_pair]
    if readings_special:
        x_avg = sum(float(r["x"]) for r in readings_special) / len(readings_special)
        y_avg = sum(float(r["y"]) for r in readings_special) / len(readings_special)
        z_avg = sum(float(r["z"]) for r in readings_special) / len(readings_special)
        averages[special_pair] = {"x": x_avg, "y": y_avg, "z": z_avg}

    for pair, avg in averages.items():
        print(f"Pair {pair}: x={avg['x']}, y={avg['y']}, z={avg['z']}")

def compose_meas_matrix():

    x_values = [(float(r["x"]) *  9.81) for r in data_log]
    y_values = [(float(r["y"]) * 9.81) for r in data_log]
    z_values = [(float(r["z"]) * 9.81) for r in data_log]

    meas_matrix = {
        "x": [[x] for x in x_values],  # a x 1 matrix for x
        "y": [[y] for y in y_values],  # a x 1 matrix for y
        "z": [[z] for z in z_values],  # a x 1 matrix for z
    }

    return meas_matrix

def compose_true_matrix(g=9.81):
    step = 15  # 15-degree increments
    max_deg = 360  # Maximum rotation for each axis pair

    true_matrix = {
        "X-Z": [],
        "X-Y": [],
        "Y-Z": []
    }

    # Transition from x to z
    # for deg in range(step, max_deg + step, step):   # start from 15 to avoid the [g,0,0] point
    for deg in range(step, max_deg + step, step):
        theta = math.radians(-deg)
        g_x = g * math.cos(theta)
        g_z = g * math.sin(theta)
        true_matrix["X-Z"].append([g_x, 0, g_z, 1])

    # Transition from y to z
    for deg in range(180 + step, max_deg + 180 + step, step):
        theta = math.radians(deg)
        g_y = g * math.cos(theta)
        g_z = g * math.sin(theta)
        true_matrix["Y-Z"].append([0, g_y, g_z, 1])

    # Transition from x to y
    for deg in range(step, max_deg + step, step):
        theta = math.radians(-deg)
        g_x = g * math.cos(theta)
        g_y = g * math.sin(theta)
        true_matrix["X-Y"].append([g_x, g_y, 0, 1])

    # Combine the matrices vertically
    combined_matrix = true_matrix["X-Z"] + true_matrix["Y-Z"] + true_matrix["X-Y"]

    return combined_matrix

def calculate_using_least_squares(M, T):

    # Convert T and M to NumPy arrays
    T = np.array(T)
    M = np.array(M).reshape(-1, 1)  # Ensure M is a column vector

    print("True Matrix (T):")
    print(T)
    print("Measured Matrix (M):")
    print(M)

    # Step 1: Compute T^T (transpose of T)
    Tt = T.T

    # Step 2: Compute T^T * T
    TtT = np.dot(Tt, T)

    # Step 3: Compute T^T * M
    TtM = np.dot(Tt, M)

    print("Shape of T:", T.shape)
    unique_rows = np.unique(T, axis=0)
    print("Number of unique rows in T:", len(unique_rows))
    

    # Step 4: Compute the inverse of T^T * T
    try:
        invTtT = np.linalg.inv(TtT)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed. Likely a faulty calibration process.")

    # Step 5: Compute x = inv(T^T * T) * T^T * M
    x = np.dot(invTtT, TtM)

    return x

def calculate_x_differences(meas_matrix, true_matrix):

    # Extract measured x-values and true x-values
    measured_x = [row[0] for row in meas_matrix["x"]]  # Flatten the measured x-matrix
    true_x = [row[0] for row in true_matrix]  # Extract x-values from the true matrix

    # Ensure the lengths match
    if len(measured_x) != len(true_x):
        raise ValueError("Measured and true x-values must have the same length.")

    # Calculate differences
    differences = [measured - actual for measured, actual in zip(measured_x, true_x)]

    # Print differences for debugging
    print("Differences (measured - actual):")
    for i, diff in enumerate(differences):
        print(f"Position {i + 1}: Difference = {diff:.4f} m/s^2")

    return differences

def calibrate_measurements(meas_matrix, result):
        # Extract measured data
    X_meas = np.array([row[0] for row in meas_matrix['x']])
    
    # Calibrate each axis separately
    ax_cal = (X_meas - result[3]) / result[0]  # (x - bx)/sxx
    
    # Stack into Nx3 array
    X_calibrated = ax_cal.reshape(-1,1)
    
    return X_calibrated

def compare_results():
    differences_before = calculate_x_differences(meas_matrix, true_matrix)
    print("Differences before calibration (x):")
    print(differences_before)

    calibrated_x = calibrate(meas_matrix, result_x)
    print (calibrated_x)
    differences = calibrated_x.flatten() - np.array([row[0] for row in true_matrix])
    print("Differences after calibration (x):")
    print(differences)

if __name__ == "__main__":
    txt_files = list_txt_files(".")
    print("Available .txt files:")
    for idx, fname in enumerate(txt_files):
        print(f"{idx}: {fname}")
    choice = int(input("Select a file by number: "))
    data_file = os.path.join(data_folder, txt_files[choice])
    read_file_data(data_file)

    true_matrix = compose_true_matrix()
    meas_matrix = compose_meas_matrix()

    result_x = calculate_using_least_squares(meas_matrix["x"], true_matrix)
    print("Calibration Result (x):")
    print(result_x)
