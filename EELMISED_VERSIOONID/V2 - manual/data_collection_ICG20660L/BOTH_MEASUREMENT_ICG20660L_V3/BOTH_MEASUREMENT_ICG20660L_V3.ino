#include <Arduino.h>1 of 35

#include "Wire.h"
#include "DFRobot_ICG20660L.h"

// -------------------- Constants --------------------
constexpr float g_exact = 9.81735f; // keep if you later convert to m/s^2
constexpr int   increment_angle = 15;
constexpr int   measurement_amount = 800;         // samples per pose
constexpr int   IMU_freq = 70;                   // Hz
constexpr int   T_init   = 120;                     // seconds, static test length
constexpr int   t_w      = 2;                     // seconds, sliding window for static detector
constexpr int   N        = 26;                    // number of manual poses (you can adjust)
constexpr int   k_thresh = 6;                     // static threshold multiplier

// -------------------- ESP32/IMU comms --------------------
const int SDA_pin = 26;
const int SCL_pin = 27;

// I2C address 0x69 (SDO high). Change to 0x68 if needed.
DFRobot_ICG20660L_IIC imu(0x69, &Wire);
sIcg20660SensorData_t accel;
sIcg20660SensorData_t gyro;
float temp = 0.0f;

volatile bool sample_ready = false;
volatile uint32_t tick_count = 0;
// =================== MANUAL OVERRIDES ===================
constexpr bool use_manual_sigma_and_bias = true;

// Manual static threshold (σ_init)
constexpr float sigma_init_m = 0.000079f;

// Manual gyro biases (units: deg/s or rad/s, depending on IMU output)
constexpr float gyro_bias_m[3] = { 2.208639f, 1.054514f, 0.363871f };

// =========================================================

// -------------------- Globals --------------------
float ax, ay, az;
float avg_ax, avg_ay, avg_az;
float gx, gy, gz;

// Static detector buffers and params(fixed-size, no heap)
constexpr int WINDOW_SIZE = t_w * IMU_freq;       // 2 s × 200 Hz = 400

static float ax_buf[WINDOW_SIZE];
static float ay_buf[WINDOW_SIZE];
static float az_buf[WINDOW_SIZE];
static int   buf_index = 0;
static bool  buffer_filled = false;
static float sigma_init = 0.0f;  // computed in test_static()

// Simple accel log struct
struct data_entry { float x, y, z; };

// Guided calib params
const int hand_num_elev = 5;      // 5 slices from -90 to +90
constexpr float angle_tolerance = 15.0f;  // degrees
int hand_total_positions;

data_entry hand_targets[N];  // enough room for generated targets

// --- Logging helpers ---
static inline void log_accel_sample(float x, float y, float z) {
  Serial.print("&X"); Serial.print(x, 4);
  Serial.print("Y");  Serial.print(y, 4);
  Serial.print("Z");  Serial.print(z, 4);
  Serial.println();
}

static inline void log_gyro_sample(float x, float y, float z) {
  Serial.print("&GX"); Serial.print(x, 4);
  Serial.print("GY");  Serial.print(y, 4);
  Serial.print("GZ");  Serial.print(z, 4);
  Serial.println();
}

// =================== TIMER VARIABLES ===================
hw_timer_t *imu_timer = NULL;        // ESP32 hardware timer
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// Global interrupt counter (increments every timer tick)
volatile uint32_t imu_counter = 0;

// Interrupt Service Routine (ISR)
void IRAM_ATTR onImuTimer() {
  tick_count++;           // count 200 Hz ticks
  sample_ready = true;    // flag: one sample due
}
// =================== STATIC TEST ===================

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
    if (!buffer_filled) return true;

    // Compute variance for each axis
    float vx = compute_variance(ax_buf, WINDOW_SIZE);
    float vy = compute_variance(ay_buf, WINDOW_SIZE);
    float vz = compute_variance(az_buf, WINDOW_SIZE);

    // Variance magnitude
    float sigma = sqrtf(vx*vx + vy*vy + vz*vz);

    return sigma <= sigma_init;
}

// void test_static() {
//   Serial.println("Running static test... keep the sensor perfectly still.");

//   constexpr int samples = T_init * IMU_freq;
//   static float ax_test[samples];
//   static float ay_test[samples];
//   static float az_test[samples];
//   // also capture gyro for bias estimation
//   float sum_gx = 0.0f, sum_gy = 0.0f, sum_gz = 0.0f;

//   // collect samples
//   for (int i = 0; i < samples; i++) {
//     read_ICG20660L();
//     ax_test[i] = ax;
//     ay_test[i] = ay;
//     az_test[i] = az;
//     sum_gx += gyro.x;
//     sum_gy += gyro.y;
//     sum_gz += gyro.z;
//     delay(1000 / IMU_freq);
//     yield();                  // let watchdog reset
//   }

//   // compute variances
//   float vx = compute_variance(ax_test, samples);
//   float vy = compute_variance(ay_test, samples);
//   float vz = compute_variance(az_test, samples);

//   // variance magnitude σ_init
//   sigma_init = k_thresh * sqrtf(vx*vx + vy*vy + vz*vz);
//   Serial.println("Static border:");
//   Serial.println(sigma_init, 6  );

//   // report gyro bias from static interval (mean per axis)
//   float bgx = sum_gx / samples;
//   float bgy = sum_gy / samples;
//   float bgz = sum_gz / samples;
//   Serial.print("&GYRO_BIAS ");
//   Serial.print("GX"); Serial.print(bgx, 6);
//   Serial.print("GY"); Serial.print(bgy, 6);
//   Serial.print("GZ"); Serial.println(bgz, 6);
//   Serial.println("&TEMP:");
//   Serial.print(temp, 2);
// }

void test_static() {
  Serial.println(F("Running static test... keep the sensor perfectly still."));

  const int samples = T_init * IMU_freq;
  float mean_ax = 0, mean_ay = 0, mean_az = 0;
  float M2_ax = 0, M2_ay = 0, M2_az = 0;  // for variance calculation
  float sum_gx = 0, sum_gy = 0, sum_gz = 0;
  int n = 0;

  // collect samples and compute mean + variance incrementally
  for (int i = 0; i < samples; i++) {
    read_ICG20660L();  // updates ax, ay, az, gx, gy, gz

    // Gyro bias accumulation
    sum_gx += gx;
    sum_gy += gy;
    sum_gz += gz;

    // Incremental variance (Welford’s method)
    n++;
    float delta_x = ax - mean_ax;
    mean_ax += delta_x / n;
    M2_ax += delta_x * (ax - mean_ax);

    float delta_y = ay - mean_ay;
    mean_ay += delta_y / n;
    M2_ay += delta_y * (ay - mean_ay);

    float delta_z = az - mean_az;
    mean_az += delta_z / n;
    M2_az += delta_z * (az - mean_az);
    yield();  // keep watchdog happy
  }

  // Compute variances
  float vx = (n > 1) ? M2_ax / (n - 1) : 0;
  float vy = (n > 1) ? M2_ay / (n - 1) : 0;
  float vz = (n > 1) ? M2_az / (n - 1) : 0;

  // Compute static threshold σ_init
  sigma_init = k_thresh * sqrtf(vx*vx + vy*vy + vz*vz);
  Serial.print(F("&STATIC"));
  Serial.println(sigma_init, 6);

  // Report gyro bias (mean over static period)
  float bgx = sum_gx / samples;
  float bgy = sum_gy / samples;
  float bgz = sum_gz / samples;

  Serial.print(F("&GYRO_BIAS "));
  Serial.print(F("GX")); Serial.print(bgx, 6);
  Serial.print(F("GY")); Serial.print(bgy, 6);
  Serial.print(F("GZ")); Serial.println(bgz, 6);
  
  Serial.print("&TEMP");
  Serial.println(temp, 2);
}

// =========================================================
void record_static_accel(float &out_ax, float &out_ay, float &out_az) {
  float sum_x = 0, sum_y = 0, sum_z = 0;
  int accepted_samples = 0;
  int meas_counter = 0;

  while (accepted_samples < measurement_amount) {
    read_ICG20660L();
    meas_counter++;

    if (!is_static() && meas_counter >= WINDOW_SIZE) {
      Serial.println("Movement detected → restarting static measurement.");
      sum_x = sum_y = sum_z = 0;
      accepted_samples = 0;
      meas_counter = 0;
      continue;
    }

    if (meas_counter < WINDOW_SIZE) {
      Serial.println("--INITIALIZING MEASUREMENT--");
    } else {
      sum_x += ax;
      sum_y += ay;
      sum_z += az;
      accepted_samples++;
      Serial.print("X"); Serial.print(ax);
      Serial.print("|Y"); Serial.print(ay);
      Serial.print("|Z"); Serial.println(az);
    }
  }

  out_ax = sum_x / measurement_amount;
  out_ay = sum_y / measurement_amount;
  out_az = sum_z / measurement_amount;

  Serial.println("--------------------------------------------------");
  log_accel_sample(out_ax, out_ay, out_az);
  Serial.print("&TEMP:");
  Serial.println(temp, 2);
  Serial.println("--------------------------------------------------");
}


int record_gyro_motion(bool if_guided, int i) {
  Serial.println("Move sensor to next pose. Recording gyro...");
  int count = 0;
  int gyro_samples = 0;

  Serial.println("&MODE=GYRO_START");

  while (is_static()) {
    read_ICG20660L(); // wait for motion
  }

  unsigned long start_time = micros();

  while (!is_static() && gyro_samples < 2000) {
    read_ICG20660L();
    log_gyro_sample(gx, gy, gz);
  

    gyro_samples++;
    count++;

    if (if_guided && !in_position(i) && gyro_samples == 1999){
      gyro_samples--;
    }

  }
  unsigned long end_us = micros();
  float elapsed = (end_us - start_time) / 1e6;
  float real_freq = gyro_samples / elapsed;
  Serial.print("&REAL_FREQ ");
  Serial.println(real_freq, 2);

  Serial.println("&MODE=GYRO_END");
  Serial.print("Recorded "); Serial.print(gyro_samples);
  Serial.println(" samples.");

  return gyro_samples;
}

void test_manual() {
  if (sigma_init == 0.0f) {
    Serial.println("Error: sigma_init not set. Run static test first!");
    while (true) delay(1000);
  }

  for (int i = 0; i < N; i++) {
    Serial.println();
    Serial.print("Position "); Serial.println(i);

    Serial.println("Recording static data...");
    record_static_accel(avg_ax, avg_ay, avg_az);

    // Serial.println("Send 'c' to continue, 'r' to redo this position.");
    // Serial.print(">");
    // while (true) {
    //   char ch = get_user_input();
    //   if (ch == 'c' || ch == 'C') break;
    //   if (ch == 'r' || ch == 'R') { i--; break; }
    // }

    record_gyro_motion(0, i);
  }
}


// =================== GUIDED CALIB ===================

float vector_angle(const data_entry& a, const data_entry& b) {
  float dot = a.x*b.x + a.y*b.y + a.z*b.z;
  float normA = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
  float normB = sqrtf(b.x*b.x + b.y*b.y + b.z*b.z);
  float c = dot / (normA * normB);
  c = constrain(c, -1.0f, 1.0f);
  return acosf(c) * 180.0f / PI;
}

bool in_position(int i){
  data_entry current = {ax, ay, az};
  float theta = vector_angle(current, hand_targets[i]);
  Serial.print("Offset from target:");
  Serial.println(theta, 1);
  return theta < angle_tolerance;
}

void generate_positions() {
  hand_total_positions = 0;

  // elevation angles in degrees (like latitude)
  int hand_elevations[hand_num_elev] = {-90, -45, 0, 45, 90};

  for (int ei = 0; ei < hand_num_elev; ei++) {
    float elev_deg = hand_elevations[ei];
    float elev_rad = elev_deg * PI / 180.0f;

    int az_steps;
    if (abs((int)elev_deg) == 90) {
      // only one point at the poles
      az_steps = 1;
    } else {
      // equator: 30 points (every 12°)
      az_steps = 8;
    }

    for (int ai = 0; ai < az_steps; ai++) {
      // use az_steps for spacing, not hand_num_az
      float az_deg = (az_steps == 1) ? 0 : ai * (360.0f / az_steps);
      float az_rad = az_deg * PI / 180.0f;

      // spherical → Cartesian
      float x = cos(elev_rad) * cos(az_rad);
      float y = cos(elev_rad) * sin(az_rad);
      float z = sin(elev_rad);

      // Prevent buffer overflow if configuration changes
      if (hand_total_positions >= (int)(sizeof(hand_targets) / sizeof(hand_targets[0]))) {
        Serial.println("Error: hand_targets capacity exceeded.");
        return;
      }
      hand_targets[hand_total_positions].x = x;
      hand_targets[hand_total_positions].y = y;
      hand_targets[hand_total_positions].z = z;
      hand_total_positions++;
    }
  }
}

void test_guided() {
  if (sigma_init == 0.0f) {
    Serial.println("Error: sigma_init not set. Run static test first!");
    while (true) delay(1000);
  }

  generate_positions();
  Serial.print("Guided calibration, total positions: ");
  Serial.println(hand_total_positions);

  if (!in_position) {
    Serial.print("Starting first guidance.");
  }

  for (int i = 0; i < hand_total_positions; i++) {
    Serial.print("Target position "); Serial.print(i+1);
    Serial.print("/"); Serial.println(hand_total_positions);

    record_static_accel(avg_ax, avg_ay, avg_az);
    record_gyro_motion(1, i);
  }

  Serial.println("Guided calibration completed.");
}

// =========================================================

// void read_ICG20660L(){
//   imu.getSensorData(&accel, &gyro, &temp);

//   // accelerometer
//   ax = accel.x;
//   ay = accel.y;
//   az = accel.z;

//   // gyroscope
//   gx = gyro.x;
//   gy = gyro.y;
//   gz = gyro.z;

//   delay(1000/IMU_freq); //depending on user chosen measuring frequency, a delay is found
// }


void read_ICG20660L() {
  // Wait until the hardware timer says a sample is ready
  while (!sample_ready) {
    // short sleep prevents busy-wait lockups
    delayMicroseconds(10);
  }

  // Clear the flag atomically
  portENTER_CRITICAL(&timerMux);
  sample_ready = false;
  portEXIT_CRITICAL(&timerMux);

  // Perform the actual IMU read (takes ~1 ms on I2C)
  imu.getSensorData(&accel, &gyro, &temp);

  ax = accel.x; ay = accel.y; az = accel.z;
  gx = gyro.x; gy = gyro.y; gz = gyro.z;
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
  Serial.begin(250000);
  while (!Serial);
  delay(1000);            // give host time to start listening

  Wire.begin(SDA_pin, SCL_pin); // SDA=21, SCL=22 for ESP32

  if (imu.begin() != 0) {
    Serial.println("Failed to initialize ICG-20660L!");
    while (1);
  }
  imu.enableSensor(0b00111111); // Bits 0-5 = gyro XYZ + accel XYZ

  Serial.println("ICG-2no0660L initialized.");

  // ---- Hardware timer for stable IMU sampling ----
  const uint32_t period_us = 1000000UL / IMU_freq;  // e.g. 5000 µs for 200 Hz
  imu_timer = timerBegin(0, 80, true);              // 80 MHz / 80 = 1 µs tick
  timerAttachInterrupt(imu_timer, &onImuTimer, true);
  timerAlarmWrite(imu_timer, period_us, true);      // periodic interrupt
  timerAlarmEnable(imu_timer);

  Serial.print("Hardware timer started at ");
  Serial.print(IMU_freq);
  Serial.println(" Hz");

  // Apply manual values if requested
  if (use_manual_sigma_and_bias) {
    Serial.print("&STATIC");
    Serial.println(sigma_init_m, 6);
    sigma_init = sigma_init_m;

    // Apply manual biases
    Serial.println("Using manual gyro biases: ");
    Serial.print(F("&GYRO_BIAS GX")); Serial.print(gyro_bias_m[0], 6);
    Serial.print(F("GY")); Serial.print(gyro_bias_m[1], 6);
    Serial.print(F("GZ")); Serial.println(gyro_bias_m[2], 6);
  }
}

void loop() {
  Serial.println("--------------------------------------------------");
  Serial.println("Choose calibration:");
  Serial.println("1 - Handheld calib");
  Serial.println("2 - Code-assisted calib");
  Serial.println("3 - Static test");
  Serial.println(">");

  char choice = get_user_input();

  if (choice == '1') {
    Serial.println("&MODE=M_ACCEL");
    test_manual();
  }
  else if (choice == '2') {
    Serial.println("&MODE=G_ACCEL");
    test_guided();
  }
  else if (choice == '3') {
    test_static();
  }
  else {
    Serial.println("Invalid choice.");
  }

} 