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

void loadGyroBias(String s) {
  const char* c = s.c_str();   // convert Arduino String → C string

  float x, y, z;

  // Read: GX<float>GY<float>GZ<float>
  if (sscanf(c, "%*s GX%fGY%fGZ%f", &x, &y, &z) == 3) {
    gyro_bias_x = x;
    gyro_bias_y = y;
    gyro_bias_z = z;
  } else {
    Serial.println("Bias parse failed!");
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
const int SERVO_PIN = 18;
const int SERVO_FREQ = 50;
const int SERVO_CH = 0;
const int SERVO_RES = 16;

const int MIN_US = 500;      // 0°
const int MAX_US = 2500;     // 360°
int offset_us = 0;

// Convert angle → PWM
void servo_goto(float angle_deg) {
  if (angle_deg < 0) angle_deg = 0;
  if (angle_deg > 360) angle_deg = 360;

  float pulse = MIN_US +
                (angle_deg / 360.0f) * (MAX_US - MIN_US)
                - offset_us;

  if (pulse < MIN_US) pulse = MIN_US;
  if (pulse > MAX_US) pulse = MAX_US;

  uint32_t duty = (uint32_t)((pulse * 65536UL) / 20000UL);
  ledcWrite(SERVO_CH, duty);
}

// ---------------- IMU READ ----------------

void readIMU() {
  while (!sample_ready) {
    delayMicroseconds(5);
  }

  portENTER_CRITICAL(&timerMux);
  sample_ready = false;
  portEXIT_CRITICAL(&timerMux);

  imu.getSensorData(&accel, &gyro, &temp);

  // -------- SUBTRACT GYRO BIAS HERE --------
  gyro.x -= gyro_bias_x;
  gyro.y -= gyro_bias_y;
  gyro.z -= gyro_bias_z;
  // -----------------------------------------
}

void clearSerial() {
  while (Serial.available()) Serial.read();
}

// ---------------- GYRO MOTION RECORDING ----------------
//
// Rotates servo by rotate_deg at speed_deg_s
// Logs gyro samples during entire motion
// Prints:
//   &MODE=GYRO_START
//   &GX...GY...GZ...
//   &MOD=GYRO_END
//
void record_gyro_rotation(float rotate_deg, float speed_deg_s) {

  Serial.println("&MODE=GYRO_START");

  float start_angle = 180;
  float end_angle = 360;

  float dt = 1.0f / IMU_FREQ;
  float step_deg = speed_deg_s * dt;

  if (step_deg < 0.143) step_deg = 0.143;
  if (step_deg > 5.0) step_deg = 5.0;

  float angle = start_angle;

  // Move to start position
  servo_goto(start_angle);
  delay(500);

  unsigned long start_time = micros();
  int sample_count = 0;

  while (angle < end_angle) {
    servo_goto(angle);
    angle += step_deg;

    readIMU();
    Serial.print("&GX"); Serial.print(gyro.x, 4);
    Serial.print("GY");  Serial.print(gyro.y, 4);
    Serial.print("GZ");  Serial.println(gyro.z, 4);

    sample_count++;
  }

  unsigned long end_time = micros();
  float freq = sample_count / ((end_time - start_time) / 1e6);

  Serial.print("&REAL_FREQ "); Serial.println(freq, 2);
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

  float angle = 360;   // you can change this if needed

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

      // Return servo to zero
      Serial.println("Centering servo...");
      servo_goto(180);
      delay(1500);

      // Stabilize IMU
      for (int k = 0; k < 10; k++) readIMU();

      // Execute run
      record_gyro_rotation(angle, speed);

      // Rest between runs
      delay(500);
    }
  }

  Serial.println("\n===== ALL 5 × 10 RUNS COMPLETE =====");
  while (1) delay(10);   // stop forever
}
