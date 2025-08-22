import serial
import threading
import time
import serial.tools.list_ports


arduino_port = 'COM6'  
baud_rate = 115200

timestamp = time.strftime('%Y%m%d-%H%M')
filename = f"calib_data_{timestamp}.txt"

print(f"Saving data to {filename}... Press Ctrl+C to stop.")

ser = None
ready_to_send = False
lock = threading.Lock()

def find_esp32_port():  # tries to automatically find the port to which the esp32 is connected
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'CP210' in port.description or 'ESP32' in port.description:  # esp32 uses cp210x driver, so script searches for that name
            return port.device  # e.g., 'COM6'
    raise Exception("ESP32 not found")

try:
    arduino_port = find_esp32_port()    # assigns port to found port
except Exception as e:
    print(f"{e}")
    arduino_port = input("Enter COM port manually (e.g., COM6): ")    # asks user to manually enter port if not found

def read_from_serial(file):
    global ready_to_send
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').rstrip() # Transforms line into utf-8 characters from bits, ignores faulty bitlines and strips newline
            if line.strip() == '>':
                print('> ', end='', flush=True) # lets user write directly after > w/o newline
            else:
                print(line)
            
            if len(line) > 0 and line[0] == '&':
                if line.startswith("&MODE="):
                    mode = line.split("=")[1]
                    file.write(f"\n--- Calibration mode: {mode} ---\n") # records the mode in which measurements made
                else:
                    file.write(line.lstrip('&') + "\n")    # writes lines marked with & into separate document, removes & symbol
                file.flush()

            if line.strip() == '>': # If arduino sends line ">", user can type
                with lock:
                    ready_to_send = True

def write_to_serial():
    global ready_to_send
    while True:
        with lock:
            if ready_to_send:
                user_input = input()
                ser.write((user_input + '\n').encode('utf-8'))
                ready_to_send = False
        time.sleep(0.1)

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to {arduino_port}")
    time.sleep(2)  # waits until Arduino initalizes

    with open(filename, 'w') as file:

        file.write(f'Calibration data from {timestamp}\n')

        reader_thread = threading.Thread(target=read_from_serial, args=(file,), daemon=True)
        writer_thread = threading.Thread(target=write_to_serial, daemon=True)

        reader_thread.start()   # two threads work simultaneusly
        writer_thread.start()

        while True:
            time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")
finally:
    if ser is not None and ser.is_open:
        ser.close()