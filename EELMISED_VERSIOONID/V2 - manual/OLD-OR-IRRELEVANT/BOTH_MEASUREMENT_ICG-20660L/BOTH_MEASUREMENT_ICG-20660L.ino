#include <Arduino.h>
#include "Wire.h"
#include "DFRobot_ICG20660L.h"

constexpr float g = 9.81735f; //exact g, may be changed
constexpr int increment_angle = 15;
constexpr int measurement_amount = 800; //measurements in one position
constexpr int pos = 360 / increment_angle; //24 different positions on each axis
constexpr int all_pos = pos*3; //all of the positions on all axes combined

//the following info is based on SER0067 servo, which is used in original testbench
constexpr int servo_pin = 18; //has to be one of the PWM pins on ESP32
constexpr int freq = 50;     //50hz servo
constexpr int channel = 0;   //PWM channel, ESP32 has 16 different channels for controlling multiple PWM outputs
constexpr int resolution = 16;  //16-bit PWM resolution (max res for max precision)
const int min_us = 500;        //min pulse for 0°
const int max_us = 2500;       //max pulse for 360°

int error_offset_us = 0;// offset from real pulse value found experimentally - this has to be subtracted from pulse value sent to servo to make it more accurate

//IMU information
const int SDA_pin = 21; //pin on ESP32
const int SCL_pin = 22;
DFRobot_ICG20660L_IIC imu(0x69, &Wire);  // Default I²C address (0x69 via SDO)
sIcg20660SensorData_t accel;
sIcg20660SensorData_t gyro;
float temp;

const int IMU_freq = 200;
const int T_init = 5;
const int t_w = 2; // in the study, a sliding time window of 2 seconds is used
const int N = 36;

constexpr int WINDOW_SIZE = t_w * IMU_freq; // 2 seconds × 200 Hz = 400 samples
float* ax_buf = new float[WINDOW_SIZE];
float* ay_buf = new float[WINDOW_SIZE];
float* az_buf = new float[WINDOW_SIZE];
int buf_index = 0;
bool buffer_filled = false;
float sigma_init = 0; // will be computed from test_static()
int const k = 6;  //threshold multiplier

//global variables for recording values over and over
float ax, ay, az;
float gx, gy, gz;

//struct for ease of use
struct data_entry {
  float x, y, z;
};

struct data_entry accel_measurement[all_pos];
struct data_entry gyro_measurement[2000][all_pos];    //max 2000 samples, ie 10 seconds moving time


void log_measurement(struct data_entry accel_measurement[], int i){
  Serial.print("&X");   //& tähistus voimaldab pythoni koodil pärast vajalikku infot eraldi faili salvestada
  Serial.print(accel_measurement[i].x, 4);  //4 komakohta
  Serial.print("Y");
  Serial.print(accel_measurement[i].y, 4);
  Serial.print("Z");
  Serial.print(accel_measurement[i].z, 4);
  Serial.println();
}

float compute_variance(float *arr, int samples) {
    if (samples < 2) return 0.0;  // avoid division by zero

    float mean = 0.0;
    for (int i = 0; i < samples; i++) {
        mean += arr[i];
    }
    mean /= samples;

    float sq = 0.0;
    for (int i = 0; i < samples; i++) {
        float diff = arr[i] - mean;
        sq += diff * diff;
    }

    return sq / (samples - 1);  // unbiased variance
}

bool is_static() {

    // Add new sample to buffer
    ax_buf[buf_index] = ax;
    ay_buf[buf_index] = ay;
    az_buf[buf_index] = az;

    buf_index = (buf_index + 1) % WINDOW_SIZE;
    if (buf_index == 0) buffer_filled = true;

    // Only start checking when buffer is full
    if (!buffer_filled) return false;

    // Compute variance for each axis
    float vx = compute_variance(ax_buf, WINDOW_SIZE);
    float vy = compute_variance(ay_buf, WINDOW_SIZE);
    float vz = compute_variance(az_buf, WINDOW_SIZE);

    // Variance magnitude
    float sigma = sqrt(vx*vx + vy*vy + vz*vz);

    return sigma <= sigma_init;
}

void test_static() {
  Serial.println("Running static test... keep the sensor perfectly still.");

  int samples = T_init * IMU_freq;
  float* ax_test = new float[samples];
  float* ay_test = new float[samples];
  float* az_test = new float[samples];

  // collect samples
  for (int i = 0; i < samples; i++) {
    imu.getSensorData(&accel, &gyro, &temp);
    ax_test[i] = accel.x;
    ay_test[i] = accel.y;
    az_test[i] = accel.z;
    delay(1000 / IMU_freq);
    yield();                  // let watchdog reset
  }

  // compute variances
  float vx = compute_variance(ax_test, samples);
  float vy = compute_variance(ay_test, samples);
  float vz = compute_variance(az_test, samples);

  // variance magnitude σ_init
  sigma_init = k * sqrt(vx*vx + vy*vy + vz*vz);
  Serial.println("Static border:");
  Serial.println(sigma_init, 6  );
}

void test_manual() {
  int gyro_counts[all_pos];

  for (int i = 0; i < N; i++) {

    Serial.println();
    Serial.println("Recording data... Please keep sensor still.");
    delay(2000);

    float sum_x = 0, sum_y = 0, sum_z = 0;
    int accepted_samples = 0;
    for (int j = 0; accepted_samples < measurement_amount; j++) {
      read_ICG20660L();

      if (!is_static()){
        Serial.println("MEAS DISCARDED - MOVEMENT");
        continue;  //if movement detected by quasi-static detector, then measurement disregarded
      }

      sum_x += ax;
      sum_y += ay;
      sum_z += az;
      accepted_samples++;
      Serial.println("ACCEPTED SAMPLES: ");
      Serial.println(accepted_samples);
    }

    accel_measurement[i].x = sum_x / measurement_amount;  //võetakse aritmeetiline keskmine kõikidest mõõtmistest ühes positsioonis, et vähendada maatriksite suurust ja minimaliseerida müra
    accel_measurement[i].y = sum_y / measurement_amount;
    accel_measurement[i].z = sum_z / measurement_amount;
  
    Serial.println("File log for position");
    Serial.print(i);
    Serial.println("--------------------------------------------------");
    log_measurement(accel_measurement, i);
    Serial.println("--------------------------------------------------");

    Serial.println("Send 'c' to continue, 'r' to calibrate again on this position.");
    Serial.print(">");

    while (true) {
      char ch = get_user_input();

      if (ch == 'c' || ch == 'C') {
        // Läheb järgmisesse positsiooni
        break;
      }
      else if (ch == 'r' || ch == 'R') {
        Serial.println("Retrying the same position...");
        i--;  // i väiksemaks, et uuesti samast positsioonist jätkata
        break;
      }
      else Serial.println(">"); 
      delay(10);
    }

    //MOTION STAGE
    Serial.println("Move sensor to next pose. Recording gyro...");
    gyro_count[i] = 0;

    while (is_static()) {
      read_ICG20660L();  // continuously feed samples
    }

    while (!is_static() && gyro_count[i] < 1000) {
      read_ICG20660L();  // reads both accel & gyro
      gyro_measurement[gyro_count][i].x = gx;
      gyro_measurement[gyro_count][i].y = gy;
      gyro_measurement[gyro_count][i].z = gz;
      
      gyro_count[i]++;
    }
    Serial.print("Recorded "); Serial.print(gyro_count[i]); Serial.println(" samples.");
  }
}

void generate_positions() {
  hand_total_positions = 0;

  // elevation angles in degrees (like latitude)
  int hand_elevations[hand_num_elev] = {-90, -45, 0, 45, 90};

  for (int ei = 0; ei < hand_num_elev; ei++) {
    float elev_deg = hand_elevations[ei];
    float elev_rad = elev_deg * PI / 180.0;

    int az_steps;
    if (abs((int)elev_deg) == 90) {
      // only one point at the poles
      az_steps = 1;
    } else {
      az_steps = hand_num_az;
    }

    for (int ai = 0; ai < az_steps; ai++) {
      float az_deg = (az_steps == 1) ? 0 : ai * (360.0 / hand_num_az);
      float az_rad = az_deg * PI / 180.0;

      // spherical → Cartesian
      float x = cos(elev_rad) * cos(az_rad);
      float y = cos(elev_rad) * sin(az_rad);
      float z = sin(elev_rad);

      hand_targets[hand_total_positions].x = x;
      hand_targets[hand_total_positions].y = y;
      hand_targets[hand_total_positions].z = z;
      hand_total_positions++;
    }
  }
}

void test_guided() {
  generate_positions();
  data_entry avg_measurement[hand_total_positions];

  Serial.print("Handheld guided calibration, total positions: ");
  Serial.println(hand_total_positions);

  for (int i = 0; i < hand_tota   l_positions; i++) {
    Serial.print("Target position "); Serial.print(i+1); Serial.print("/");
    Serial.println(hand_total_positions);

    int meas_counter = 0;
    while (true) {
      meas_counter++;

      if (in_position(i)) {

        // wait until static detector says stable
        int accepted_samples = 0;
        float sum_x=0, sum_y=0, sum_z=0;

        Serial.println("Type c when sensor in a stable position OR anything else to check offset.");
        while(1){
          if (in_position(i)) Serial.println("In position.");
          Serial.println(">");
          char input = get_user_input();
          if (input == 'c' || input == 'C') break;
        }
        
        while (accepted_samples < measurement_amount) {
          read_ICG20660L();
          if (!is_static() && meas_counter >= 425) {
            Serial.println("Movement detected → restarting");
            meas_counter = 0;
            continue;
          }
          else {
            if (meas_counter < 425) {
              Serial.println("--INITIALIZING MEASUREMENT--");
              Serial.println(meas_counter);
              }
            else {
            sum_x += ax_g;
            sum_y += ay_g;
            sum_z += az_g;
            accepted_samples++;
            }
            meas_counter++;
          }
        }

        avg_measurement[i].x = sum_x / measurement_amount;
        avg_measurement[i].y = sum_y / measurement_amount;
        avg_measurement[i].z = sum_z / measurement_amount;

        log_measurement(avg_measurement, i);
        Serial.println(accepted_samples);
        break; // move to next target
      }
      else {
        Serial.print("Not aligned...");
      }
      delay(200);
    }
  }
  Serial.println("Guided handheld calibration completed.");
}

void test_servo() {
  struct data_entry accel_measurement[all_pos];
  
  for (int axis = 0; axis < 3; axis++){
    rotate_servo(0);  //initially rotate servo to starting position, so no jerky movement when calibration starts

    Serial.println("Type 'c' and press ENTER to start calibration on this axis.");
    char input;
    do {
      Serial.println(">");
      input = get_user_input();
    } while (input != 'c' && input != 'C');
    Serial.println("--------------------------------------------------");

    for (int i = 1; i < (pos + 1); i++) { //position 0 is ignored, because ser0067 cannot accurately hold that position
      rotate_servo(i);  //servo rotates to next position depending on i, which cycles through all positions on an axis
      delay(500); // waiting time to ensure that measuring platform settles

      float sum_x = 0, sum_y = 0, sum_z = 0;
      int accepted_samples = 0;   // variable needed because of the static detector
      for (int j = 0; accepted_samples < measurement_amount; j++) { //starts measuring in this position
        read_ICG20660L();

        if (!is_static()){
          Serial.println("MEAS DISCARDED - MOVEMENT");
          continue;  //if movement detected by quasi-static detector, then measurement disregarded
        }

        sum_x += ax;
        sum_y += ay;
        sum_z += az;
        accepted_samples++;

      }

      accel_measurement[i].x = sum_x / accepted_samples;  //average is taken to minimize noise influence on calib results
      accel_measurement[i].y = sum_y / accepted_samples;
      accel_measurement[i].z = sum_z / accepted_samples;

      log_measurement(accel_measurement, i);
      Serial.println("--------------------------------------------------");
    }
    if (axis != 2) {
      Serial.println("Entering debug to calibrate before next axis:");
      debug();
      Serial.print("Change axis! Next axis: ");
      }
    else Serial.println("Calibration completed succesfully.");

    if (axis == 0) Serial.println("Y- axis, blue marking.");
    if (axis == 1) Serial.println("Z- axis, green marking.");
  }
}

void debug(){   //code for checking angles etc. by allowing user to enter any angle for calibration
  while (1) {
    Serial.println(">");
    while (!Serial.available()) delay(10); // Wait until user inputs something
    String input = Serial.readStringUntil('\n'); // Read string by user
    input.trim();

    int pulse_us = -1;

    // Check if angle command
    if (input.startsWith("A")) {
      int angle = input.substring(1).toInt();
      if (angle < 0) angle = 0;
      if (angle > 360) angle = 360;

      // Map angle to pulse, add 10 µs to correct offset
      pulse_us = min_us + ((long)(max_us - min_us) * angle) / 360 - error_offset_us;

    } 
    else if (input.startsWith("E")) {
      error_offset_us = input.substring(1).toInt();
      continue;
    } 
    else if (input == "0") break;
    else pulse_us = input.toInt(); // Direct pulse command

    // Clamp pulse within safe range
    if (pulse_us < 500) pulse_us = 500;
    if (pulse_us > 2500) pulse_us = 2500;

    // Convert pulse to duty for 16-bit PWM
    uint32_t duty = (uint32_t)((pulse_us * 65536UL) / 20000UL);
    ledcWrite(servo_pin, duty);

    Serial.print("Moved to pulse: ");
    Serial.println(pulse_us);
  }
}

void rotate_servo(int current_pos){

  int angle = current_pos * increment_angle;
  float pulse = min_us + (angle / 360.0) * (max_us - min_us) - error_offset_us;  // Convert angle to pulse width (µs)

  // Clamp pulse within safe range
  if (pulse < min_us) pulse = min_us;
  if (pulse > max_us) pulse = max_us; 

  // Convert pulse width (µs) to duty cycle
  // - divide by servo period (20000 µs = 20 ms for 50 Hz)
  // - scale to 16-bit resolution (0–65535) -> also why uint32_t used
  uint32_t duty = (uint32_t)((pulse * 65536UL) / 20000UL);
  // Write duty cycle to servo
  ledcWrite(servo_pin, duty);
}

void read_ICG20660L(){
  imu.getSensorData(&accel, &gyro, &temp);

  // accelerometer
  ax = accel.x;
  ay = accel.y;
  az = accel.z;

  // gyroscope
  gx = gyro.x;
  gy = gyro.y;
  gz = gyro.z;

  Serial.print("Temp: "); Serial.println(temp, 2);

  delay(1000/IMU_freq); //depending on user chosen measuring frequency, a delay is found
}

char get_user_input(){
  // Clear any leftover input
  while (Serial.available() > 0) {
    Serial.read();
  }
  
  // Wait until user inputs something
  while (!Serial.available()) {
    delay(10);
  }
  
  // Read first char
  char ch = Serial.read();
  
  // Clear any extra chars user might have typed
  while (Serial.available() > 0) {
    Serial.read();
  }
  
  return ch;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);
  delay(1000);            // give host time to start listening

  Wire.begin(SDA_pin, SCL_pin); // SDA=21, SCL=22 for ESP32

  ledcAttach(servo_pin, freq, resolution);

  if (imu.begin() != 0) {
    Serial.println("Failed to initialize ICG-20660L!");
    while (1);
  }
  imu.enableSensor(0b00111111); // Bits 0-5 = gyro XYZ + accel XYZ

  Serial.println("ICG-20660L initialized.");
}

void loop() {
  Serial.println("--------------------------------------------------");
  Serial.println("Choose calibration:");
  Serial.println("1 - Handheld calib");
  Serial.println("2 - Code-assisted calib");
  Serial.println("3 - Servo-assisted calib");
  Serial.println("4 - Static test");
  Serial.println("5 - Debug");
  Serial.println(">");

  char choice = get_user_input();

  if (choice == '1') {
    Serial.println("&MODE=M_ACCEL");
    test_manual();
  }
  else if (choice == '2') {
    Serial.println("&MODE=S_ACCEL");
    test_servo();
  }
  else if (choice == '3') {
    test_static();
  }
  else if (choice == '4') {
    debug();
  }
  else {
    Serial.println("Invalid choice.");
  }

} 