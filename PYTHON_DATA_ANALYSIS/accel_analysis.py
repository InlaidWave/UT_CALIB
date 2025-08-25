import os

mode = None
data_log = []

def list_txt_files(directory, subfolder="DATA"):
    folder_path = os.path.join(directory, subfolder)
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

def read_file_data(file_path):
    data_log_position = -1

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("MODE="):
                mode_f = line[len("MODE="):]  # Gets everything after 'MODE='
                if mode_f == "M_ACCEL":
                    mode = "Manual testbench accelerometer calibration"
                elif mode_f == "S_ACCEL":
                    mode == "Motorized testbench accelerometer calibration"
                else:
                    mode = mode_f   # makes mode raw mode name if not recognised
                print("Calibration mode:", mode)
            else:
                data_log_position += 1
                line = line.strip()
                x_start = line.find('X') + 1
                y_start = line.find('Y') + 1
                z_start = line.find('Z') + 1

                x_value = line[x_start:line.find('Y')]
                y_value = line[y_start:line.find('Z')]
                z_value = line[z_start:line.find('A')]
                axis = line[line.find('A') + 1]

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

def compose_true_matrices():
    pass

def compose_meas_matrices():
    pass

def calculate__using_least_squares():
    pass

if __name__ == "__main__":
    txt_files = list_txt_files(".")
    print("Available .txt files:")
    for idx, fname in enumerate(txt_files):
        print(f"{idx}: {fname}")
    choice = int(input("Select a file by number: "))
    data_file = txt_files[choice]
    read_file_data(data_file)