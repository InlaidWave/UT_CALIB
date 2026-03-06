/*
  imu_record_no_init.ino
  ESP32 sketch to record ICG20660L gyro data for Allan analysis.

  Behavior:
   - Wait for ENTER on Serial.
   - Wait 5 seconds (delay for operator).
   - Stream raw gyro samples (no bias subtraction) in format:
       &T<micros>&GX<gx>GY<gy>GZ<gz>
   - RECORD_DURATION_SEC controls how long to record (0 = infinite).
*/

#include <Arduino.h>
#include "Wire.h"
#include "DFRobot_ICG20660L.h"

// ---------- CONFIG ----------
const int SDA_PIN = 26;
const int SCL_PIN = 27;
const uint8_t IMU_ADDR = 0x69;

const int IMU_FREQ = 70;                    // Hz sampling rate (timer driven)
const unsigned long PRE_RECORD_DELAY_MS = 15000; // 5 seconds before recording starts
const unsigned long RECORD_DURATION_SEC = 3600;  // seconds to record after delay; 0 = infinite

// Serial baud
const uint32_t SERIAL_BAUD = 250000;

// ---------- Globals ----------
DFRobot_ICG20660L_IIC imu(IMU_ADDR, &Wire);

sIcg20660SensorData_t accel;
sIcg20660SensorData_t gyro;
float temp_c;

// Timer flag
volatile bool sample_ready = false;
hw_timer_t* imu_timer = nullptr;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// ISR for timer
void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  sample_ready = true;
  portEXIT_CRITICAL_ISR(&timerMux);
}

// ---------- Setup & helpers ----------
void setupIMU() {
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(10);
  if (!imu.begin()) {
    Serial.println("IMU begin failed!");
  }
  imu.enableSensor(0b00111111); // enable accel + gyro + temp
}

// Read a single IMU sample (blocks until sample_ready)
void readIMU_blocking() {
  while (!sample_ready) {
    delayMicroseconds(50);
  }
  portENTER_CRITICAL(&timerMux);
  sample_ready = false;
  portEXIT_CRITICAL(&timerMux);

  imu.getSensorData(&accel, &gyro, &temp_c);
}

// Print one sample line in format &T<micros>&GX...GY...GZ...
void print_sample_line(float gx, float gy, float gz) {
  unsigned long ts = micros();
  Serial.print("&T"); Serial.print(ts);
  Serial.print("&GX"); Serial.print(gx, 6);
  Serial.print("GY"); Serial.print(gy, 6);
  Serial.print("GZ"); Serial.println(gz, 6);
}

// ---------- Arduino setup ----------
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500); // allow host to connect
  while (!Serial) { delay(10); } // wait for serial port (USB)
  Serial.println("IMU recorder (no init). Press ENTER to start; 5s delay before capture.");

  setupIMU();

  // Setup hardware timer for IMU sampling
  imu_timer = timerBegin(0, 80, true);
  timerAttachInterrupt(imu_timer, &onTimer, true);
  timerAlarmWrite(imu_timer, 1000000UL / IMU_FREQ, true); // period in microseconds
  timerAlarmEnable(imu_timer);

  Serial.print("IMU freq = "); Serial.print(IMU_FREQ); Serial.println(" Hz");
  Serial.print("Pre-record delay = "); Serial.print(PRE_RECORD_DELAY_MS / 1000); Serial.println(" s");
  if (RECORD_DURATION_SEC > 0) {
    Serial.print("Recording duration = "); Serial.print(RECORD_DURATION_SEC); Serial.println(" s");
  } else {
    Serial.println("Recording duration = infinite (stop manually)");
  }
  Serial.println(">");
}

// ---------- Arduino loop ----------
void loop() {
  // wait for user to press ENTER
  while (Serial.available() == 0) {
    delay(10);
  }
  // clear input buffer
  while (Serial.available()) Serial.read();
  delay(10);

  // short notice, then delay PRE_RECORD_DELAY_MS
  Serial.println("Starting capture in 5 seconds...");
  unsigned long wait_start = millis();
  while (millis() - wait_start < PRE_RECORD_DELAY_MS) {
    // optional: print a dot each second (comment out if you want totally clean log)
    // if ((millis() - wait_start) % 1000 < 20) Serial.print('.');
    delay(10);
  }
  Serial.println(); // newline after the delay countdown

  // Start recording
  Serial.println("&MODE=RECORD_START");
  unsigned long record_start_ms = millis();
  unsigned long recorded_samples = 0;

  while (true) {
    // read next sample (blocks until IMU timer tick)
    readIMU_blocking();

    // print raw gyro values (no bias subtraction)
    print_sample_line(gyro.x, gyro.y, gyro.z);
    recorded_samples++;

    // check duration if finite
    if (RECORD_DURATION_SEC > 0) {
      unsigned long elapsed_s = (millis() - record_start_ms) / 1000UL;
      if (elapsed_s >= RECORD_DURATION_SEC) {
        break;
      }
    }

    // keep loop tight; do not delay more than sample period
  }

  Serial.println("&MODE=RECORD_END");
  Serial.print("Recorded samples: "); Serial.println(recorded_samples);
  Serial.println("Capture finished. Press ENTER to start another run...");
  // loop will wait for ENTER again
}