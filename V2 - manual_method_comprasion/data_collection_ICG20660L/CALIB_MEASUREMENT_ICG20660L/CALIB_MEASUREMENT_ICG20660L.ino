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
float sigma_init = 0; // computed from test_static()
int const k = 6;  //threshold multiplier

//global variables for recording values over and over
float ax_g, ay_g, az_g;

//struct for ease of use
struct data_entry {
  float x, y, z;
};


void log_measurement(struct data_entry avg_measurement[], int i){
  Serial.print("&X");   //& tähistus voimaldab pythoni koodil pärast vajalikku infot eraldi faili salvestada
  Serial.print(avg_measurement[i].x, 4);  //4 komakohta
  Serial.print("Y");
  Serial.print(avg_measurement[i].y, 4);
  Serial.print("Z");
  Serial.print(avg_measurement[i].z, 4);
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
    ax_buf[buf_index] = ax_g;
    ay_buf[buf_index] = ay_g;
    az_buf[buf_index] = az_g;

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
  struct data_entry avg_measurement[all_pos];

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

      sum_x += ax_g;
      sum_y += ay_g;
      sum_z += az_g;
      accepted_samples++;
      Serial.println("ACCEPTED SAMPLES: ");
      Serial.println(accepted_samples);
    }

    avg_measurement[i].x = sum_x / measurement_amount;  //võetakse aritmeetiline keskmine kõikidest mõõtmistest ühes positsioonis, et vähendada maatriksite suurust ja minimaliseerida müra
    avg_measurement[i].y = sum_y / measurement_amount;
    avg_measurement[i].z = sum_z / measurement_amount;
  
    Serial.print("Average X: "); Serial.print(avg_measurement[i].x, 4);
    Serial.print(" Y: "); Serial.print(avg_measurement[i].y, 4);
    Serial.print(" Z: "); Serial.println(avg_measurement[i].z, 4);

    if (i != (all_pos - 1)){    
      Serial.println("File log for position");
      Serial.print(i);
      Serial.println("--------------------------------------------------");
      log_measurement(avg_measurement, i);
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
    }
  }
}

void test_servo() {
  struct data_entry avg_measurement[all_pos];
  
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

        sum_x += ax_g;
        sum_y += ay_g;
        sum_z += az_g;
        accepted_samples++;

      }

      avg_measurement[i].x = sum_x / accepted_samples;  //average is taken to minimize noise influence on calib results
      avg_measurement[i].y = sum_y / accepted_samples;
      avg_measurement[i].z = sum_z / accepted_samples;

      log_measurement(avg_measurement, i);
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

  // But only use accel
  Serial.print("Accel X: "); Serial.print(accel.x, 4);
  ax_g = accel.x;

  Serial.print(" g  Y: ");   Serial.print(accel.y, 4);
  ay_g = accel.y;

  Serial.print(" g  Z: ");   Serial.println(accel.z, 4);
  az_g = accel.z;

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
  Serial.println("2 - Servo-assisted calib");
  Serial.println("3 - Static test");
  Serial.println("4 - Debug");
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