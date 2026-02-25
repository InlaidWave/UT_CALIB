#include <Arduino.h>
#include "Wire.h"
#include "DFRobot_ICG20660L.h"

// ---------------- IMU SETTINGS ----------------
const int SDA_PIN = 26;
const int SCL_PIN = 27;
const int IMU_ADDR = 0x69;
const int IMU_FREQ = 70;     // Hz (gyro sample rate)

float gyro_bias_x = 0;
float gyro_bias_y = 0;
float gyro_bias_z = 0;

String BIAS_STRING = "&GYRO_BIAS GX2.207271GY1.248071GZ0.380602";

void loadGyroBias(const String &s) {
  // Robustly find GX, GY, GZ tokens and parse floats after them.
  const char* c = s.c_str();
  const char* px = strstr(c, "GX");
  const char* py = strstr(c, "GY");
  const char* pz = strstr(c, "GZ");
  if (px && py && pz) {
    // use atof on substring after token
    gyro_bias_x = atof(px + 2);
    gyro_bias_y = atof(py + 2);
    gyro_bias_z = atof(pz + 2);
  } else {
    Serial.println("Bias parse failed (tokens not found)!");
  }
}

DFRobot_ICG20660L_IIC imu(IMU_ADDR, &Wire);
sIcg20660SensorData_t accel;
sIcg20660SensorData_t gyro;
float temp;

// Timer interrupt flag
volatile bool sample_ready = false;

// Hardware timer
hw_timer_t* imu_timer = nullptr;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// ISR (fires at IMU_FREQ)
void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  sample_ready = true;
  portEXIT_CRITICAL_ISR(&timerMux);
}

// ---------------- SERVO SETTINGS ----------------
const int SERVO_PIN = 25;
const int SERVO_FREQ = 50;
const int SERVO_CH = 0;
const int SERVO_RES = 16;

const int MIN_US = 500;      // 0°
const int MAX_US = 2500;     // 360°
int offset_us = 0;
const unsigned long SERVO_PERIOD_US = 1000000UL / SERVO_FREQ;

// Convert angle → PWM (clamped 0..360). Uses correct duty scaling for ESP32 ledc.
void servo_goto(float angle_deg) {
  if (angle_deg < 0) angle_deg = 0;
  if (angle_deg > 360) angle_deg = 360;

  float pulse = MIN_US + (angle_deg / 360.0f) * (MAX_US - MIN_US) - offset_us;

  if (pulse < MIN_US) pulse = MIN_US;
  if (pulse > MAX_US) pulse = MAX_US;

  // duty range is 0 .. (2^RES - 1)
  uint32_t max_duty = (1UL << SERVO_RES) - 1UL;
  uint32_t duty = (uint32_t)((pulse * (float)max_duty) / (float)SERVO_PERIOD_US);
  if (duty > max_duty) duty = max_duty;

  ledcWrite(SERVO_CH, duty);
}

// ---------------- IMU READ ----------------
void readIMU() {
  // Wait for timer/IMU tick
  while (!sample_ready) {
    // small yield to avoid tight busy spin
    delayMicroseconds(50);
  }

  portENTER_CRITICAL(&timerMux);
  sample_ready = false;
  portEXIT_CRITICAL(&timerMux);

  imu.getSensorData(&accel, &gyro, &temp);

  // subtract gyro bias
  gyro.x -= gyro_bias_x;
  gyro.y -= gyro_bias_y;
  gyro.z -= gyro_bias_z;
}

void clearSerial() {
  while (Serial.available()) Serial.read();
}

// ---------------- GYRO MOTION RECORDING ----------------
//
// Rotates servo by rotate_deg at speed_deg_s (deg/s)
// Logs gyro samples during entire motion
//
void record_gyro_rotation(float rotate_deg, float speed_deg_s) {

  Serial.println("&MODE=GYRO_START");

  // Use a centered starting position so we can rotate CW or CCW as needed.
  const float home_angle = 180.0f;        // start from mid-position
  float start_angle = home_angle;
  float end_angle = home_angle + rotate_deg; // rotate positive direction

  // Bound rotate_deg to avoid exceeding servo limits
  if (end_angle > 360.0f) {
    end_angle = 360.0f;
  }

  float dt = 1.0f / (float)IMU_FREQ;
  float step_deg = speed_deg_s * dt;   // degrees per IMU sample

  // ensure step is not zero; clamp to reasonable min/max
  if (step_deg < 0.1f) step_deg = 0.1f;
  if (step_deg > 10.0f) step_deg = 10.0f;

  // Move to start position and allow settle
  servo_goto(start_angle);
  delay(500);

  unsigned long start_time = micros();
  int sample_count = 0;
  float angle = start_angle;

  // Pre-warm a few IMU samples so measurement stabilizes before motion
  for (int k = 0; k < 5; ++k) readIMU();

  // Main loop: for each expected sample, command the servo then readIMU()
  while (angle < end_angle) {
    // command next angle (do not exceed end_angle)
    float next_angle = angle + step_deg;
    if (next_angle > end_angle) next_angle = end_angle;

    servo_goto(next_angle);

    // Wait for the next IMU sample and read it
    readIMU();

    // Print gyro sample (timestamp optional)
    unsigned long ts = micros();
    Serial.print("&T"); Serial.print(ts);
    Serial.print("&GX"); Serial.print(gyro.x, 4);
    Serial.print("GY");  Serial.print(gyro.y, 4);
    Serial.print("GZ");  Serial.println(gyro.z, 4);

    sample_count++;
    angle = next_angle;
  }

  unsigned long end_time = micros();
  float elapsed_s = (end_time - start_time) / 1e6f;
  float real_freq = sample_count / elapsed_s;

  Serial.print("&REAL_FREQ "); Serial.println(real_freq, 2);
  Serial.println("&MODE=GYRO_END");
  Serial.print("Recorded "); Serial.print(sample_count);
  Serial.println(" gyro samples.");
}

// ---------------- SETUP ----------------

void setup() {
  Serial.begin(250000);
  delay(500);  // IMPORTANT: give USB time to enumerate
  while (!Serial) {
      delay(10);
  }
  Serial.println("Serial connected!");

  // Servo setup
  ledcSetup(SERVO_CH, SERVO_FREQ, SERVO_RES);
  ledcAttachPin(SERVO_PIN, SERVO_CH);
  servo_goto(180); // safe center

  // IMU setup
  Wire.begin(SDA_PIN, SCL_PIN);
  imu.begin();
  imu.enableSensor(0b00111111);

  // Timer for stable IMU sampling
  imu_timer = timerBegin(0, 80, true);
  timerAttachInterrupt(imu_timer, &onTimer, true);
  timerAlarmWrite(imu_timer, 1000000UL / IMU_FREQ, true);
  timerAlarmEnable(imu_timer);

  loadGyroBias(BIAS_STRING);

  Serial.print("Bias X = "); Serial.println(gyro_bias_x, 6);
  Serial.print("Bias Y = "); Serial.println(gyro_bias_y, 6);
  Serial.print("Bias Z = "); Serial.println(gyro_bias_z, 6);

  Serial.println("System ready.");
}

// ---------------- LOOP ----------------

void loop() {

  float angle = 180.0f;   // rotate span in degrees (use rotate_deg rather than absolute end pos)

  Serial.println("Press ENTER to start...");
  Serial.print(">");
  clearSerial();
  while (Serial.available() == 0) {
      delay(10);
  }
  clearSerial();
  Serial.println();
  Serial.println("Starting now!");

  for (int repeat = 1; repeat <= 5; repeat++) {
    Serial.print("\n=== REPEAT SET ");
    Serial.print(repeat);
    Serial.println(" / 5 ===");

    for (int i = 0; i < 10; i++) {

      float speed = 10 + i * 10;     // 10, 20, 30, ..., 100 deg/s

      Serial.println("\n------------------------------------");
      Serial.print("&R"); Serial.print(i+1);
      Serial.print("S"); Serial.print(repeat);
      Serial.print("V"); Serial.println(speed);
      Serial.println("------------------------------------");

      // Return servo to center
      Serial.println("Centering servo...");
      servo_goto(180);
      delay(1500);

      // Stabilize IMU
      for (int k = 0; k < 10; k++) readIMU();

      // Execute run: rotate by `angle` degrees at `speed`
      record_gyro_rotation(angle, speed);

      // Rest between runs
      delay(500);
    }
  }

  Serial.println("\n===== ALL 5 × 10 RUNS COMPLETE =====");
  while (1) delay(10);   // stop forever
}