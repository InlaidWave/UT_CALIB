#include <Arduino.h>
#include "Wire.h"
#include "DFRobot_ICG20660L.h"

// -------------------- Constants --------------------
constexpr float g_exact = 9.81735f; // keep if you later convert to m/s^2
constexpr int   increment_angle = 15;
constexpr int   measurement_amount = 800;         // samples per pose
constexpr int   IMU_freq = 200;                   // Hz
constexpr int   T_init   = 60;                     // seconds, sliding window for static detector
constexpr int   t_w      = 2;                     // seconds, sliding window for static detector
constexpr int   N        = 38;                    // number of manual poses (you can adjust)
constexpr int   k_thresh = 6;                     // static threshold multiplier

// -------------------- ESP32/IMU comms --------------------
const int SDA_pin = 26;
const int SCL_pin = 27;

// I2C address 0x69 (SDO high). Change to 0x68 if needed.
DFRobot_ICG20660L_IIC imu(0x69, &Wire);
sIcg20660SensorData_t accel;
sIcg20660SensorData_t gyro;
float temp = 0.0f;

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

//---code---

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

    delay(1000 / IMU_freq);
    yield();  // keep watchdog happy
  }

  // Compute variances
  float vx = (n > 1) ? M2_ax / (n - 1) : 0;
  float vy = (n > 1) ? M2_ay / (n - 1) : 0;
  float vz = (n > 1) ? M2_az / (n - 1) : 0;

  // Compute static threshold σ_init
  sigma_init = k_thresh * sqrtf(vx*vx + vy*vy + vz*vz);
  Serial.println(F("Static border:"));
  Serial.println(sigma_init, 6);

  // Report gyro bias (mean over static period)
  float bgx = sum_gx / samples;
  float bgy = sum_gy / samples;
  float bgz = sum_gz / samples;

  Serial.print(F("&GYRO_BIAS "));
  Serial.print(F("GX")); Serial.print(bgx, 6);
  Serial.print(F("GY")); Serial.print(bgy, 6);
  Serial.print(F("GZ")); Serial.println(bgz, 6);
  
  Serial.println("&TEMP:");
  Serial.print(temp, 2);
}

void test_manual() {
  int gyro_counts[N];

  if (sigma_init == 0.0f) {
    Serial.println("Error: sigma_init not set. Run static test first!");
    while (true) delay(1000); // halt safely
  }

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
      Serial.println("X");
      Serial.print(ax);
      Serial.print("|Y");
      Serial.print(ay);
      Serial.print("|Z");
      Serial.print(az);
    }

    avg_ax = sum_x / measurement_amount;  //to minimize noise take average of all readings in a position
    avg_ay = sum_y / measurement_amount;
    avg_az = sum_z / measurement_amount;
  
    Serial.println("File log for position");
    Serial.print(i);
    Serial.println("--------------------------------------------------");
    log_accel_sample(avg_ax, avg_ay, avg_az);
    Serial.println("&TEMP:");
    Serial.print(temp, 2);
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
  gyro_counts[i] = 0;
  Serial.println("&MODE=GYRO_START");

    while (is_static()) {
      read_ICG20660L();  // continuously feed samples
    }

    while (!is_static() && gyro_counts[i] < 1000) {
      read_ICG20660L();  // reads both accel & gyro

      // Log each gyro sample for off-board processing
      log_gyro_sample(gx, gy, gz);

      gyro_counts[i]++;
    }
    Serial.println("&MODE=GYRO_END");
    Serial.print("Recorded "); Serial.print(gyro_counts[i]); Serial.println(" samples.");
  }
}

float vector_angle(const data_entry& a, const data_entry& b) {
  float dot = a.x*b.x + a.y*b.y + a.z*b.z;
  float normA = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
  float normB = sqrtf(b.x*b.x + b.y*b.y + b.z*b.z);
  float c = dot / (normA * normB);
  c = constrain(c, -1.0f, 1.0f);
  return acosf(c) * 180.0f / PI;
}

bool in_position(int i){
  read_ICG20660L();
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
      az_steps = 30;
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
    while (true) delay(1000); // halt safely
  }

  generate_positions();
  static data_entry avg_measurement[N]; // capped to N (== hand_targets size)

  Serial.print("Handheld guided calibration, total positions: ");
  Serial.println(hand_total_positions);

  for (int i = 0; i < hand_total_positions; i++) {
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
            sum_x += ax;
            sum_y += ay;
            sum_z += az;
            accepted_samples++;

            Serial.println("X");
            Serial.print(ax);
            Serial.print("|Y");
            Serial.print(ay);
            Serial.print("|Z");
            Serial.print(az);
            }
            meas_counter++;
          }
        }

        avg_ax = sum_x / measurement_amount;
        avg_ay = sum_y / measurement_amount;
        avg_az = sum_z / measurement_amount;

        Serial.println("File log for position");
        Serial.print(i);
        Serial.println("--------------------------------------------------");
        log_accel_sample(avg_ax, avg_ay, avg_az);
        Serial.println("--------------------------------------------------");

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