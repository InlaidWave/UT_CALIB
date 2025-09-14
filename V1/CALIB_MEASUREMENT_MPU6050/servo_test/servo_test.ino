#include "Wire.h"
#include "MPU6050.h"

MPU6050 mpu;

// SER0067 corrected pulse/angle control for ESP32
//MPU 6050-servo system debug

const int servoPin = 18;      // PWM-capable pin connected to servo signal
const int ledChannel = 0;
const int freq = 50;          // 50 Hz for standard servo
const int resolution = 16;    // 16-bit

const int minUs = 500;        // min pulse for 0°
const int maxUs = 2500;       // max pulse for 360°

const int SDA_pin = 21;
const int SCL_pin = 22;


void read_MPU6050(){
  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert to real units
  float ax_g = ax / 16384.0;  // g
  float ay_g = ay / 16384.0;
  float az_g = az / 16384.0;

  float gx_dps = gx / 131.0;  // deg/sec
  float gy_dps = gy / 131.0;
  float gz_dps = gz / 131.0;

  Serial.print("Accel [g]:\t");
  Serial.print(ax_g); Serial.print("\t");
  Serial.print(ay_g); Serial.print("\t");
  Serial.print(az_g); Serial.print("\t");

  delay(500);
}

void setup() {
  Serial.begin(115200);
  ledcAttach(servoPin, freq, resolution);
  Serial.println("Enter pulse (µs) or angle (0-360) prefixed with A:");
  Serial.println("Examples: 1500  or  A90");

  Wire.begin(SDA_pin, SCL_pin); // SDA, SCL on ESP32
  mpu.initialize();
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int pulseUs = -1;

    // Check if angle command
    if (input.startsWith("A")) {
      int angle = input.substring(1).toInt();
      if (angle < 0) angle = 0;
      if (angle > 360) angle = 360;

      // Map angle to pulse, add 10 µs to correct offset
      pulseUs = minUs + ((long)(maxUs - minUs) * angle) / 360 - 13;

    } else {
      // Direct pulse command
      pulseUs = input.toInt();
    }

    // Clamp pulse within safe range
    if (pulseUs < 500) pulseUs = 500;
    if (pulseUs > 2500) pulseUs = 2500;

    // Convert pulse to duty for 16-bit PWM
    uint32_t duty = (uint32_t)((pulseUs * 65536UL) / 20000UL);
    ledcWrite(servoPin, duty);

    Serial.print("Moved to pulse: ");
    Serial.println(pulseUs);
  }
  read_MPU6050();
}