#include <Arduino.h>
#include "Wire.h"
#include "DFRobot_ICG20660L.h"

// ---------------- IMU SETTINGS ----------------
const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int IMU_ADDR = 0x69;
const int IMU_FREQ = 200;     // Hz (gyro sample rate)

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

  float start_angle = 0;
  float end_angle = rotate_deg;

  float dt = 1.0f / IMU_FREQ;
  float step_deg = speed_deg_s * dt;

  if (step_deg < 0.5) step_deg = 0.5;
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

  // Final IMU samples after stop
  for (int i = 0; i < 50; i++) {
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
  delay(1000);

  // Servo setup
  ledcAttach(SERVO_PIN, SERVO_FREQ, SERVO_RES);

  // IMU setup
  Wire.begin(SDA_PIN, SCL_PIN);
  imu.begin();
  imu.enableSensor(0b00111111);

  // Timer for stable IMU sampling
  imu_timer = timerBegin(0, 80, true);
  timerAttachInterrupt(imu_timer, &onTimer, true);
  timerAlarmWrite(imu_timer, 1000000UL / IMU_FREQ, true);
  timerAlarmEnable(imu_timer);

  Serial.println("System ready.");
}


// ---------------- LOOP ----------------

void loop() {
  Serial.println("Enter rotation angle (deg):");
  while (!Serial.available()) delay(10);

  float angle = Serial.parseFloat();

  Serial.println("Enter speed (deg/s):");
  while (!Serial.available()) delay(10);

  float speed = Serial.parseFloat();

  Serial.print("Rotating "); Serial.print(angle);
  Serial.print(" degrees at "); Serial.print(speed); Serial.println(" deg/s");

  record_gyro_rotation(angle, speed);
}
