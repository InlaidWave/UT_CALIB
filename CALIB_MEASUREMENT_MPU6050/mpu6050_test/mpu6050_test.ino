#include "Wire.h"
#include "MPU6050.h"

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_pin, SCL_pin); // SDA, SCL on ESP32
  mpu.initialize(); 

  Serial.println("Initializing MPU6050...");
  mpu.initialize();

  if (mpu.testConnection()) {
    Serial.println("MPU6050 connection successful!");
  } else {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }
}

void loop() {
  read_MPU6050()
}

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

  Serial.print(" Gyro [deg/s]:\t");
  Serial.print(gx_dps); Serial.print("\t");
  Serial.print(gy_dps); Serial.print("\t");
  Serial.println(gz_dps);

  delay(500);
}