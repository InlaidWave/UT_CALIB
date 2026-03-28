#include <Wire.h>
#include "DFRobot_ICG20660L.h"

DFRobot_ICG20660L_IIC imu(0x69, &Wire);  // Default I²C address (0x68 or 0x69 via SDO)

sIcg20660SensorData_t accel;
sIcg20660SensorData_t gyro;
float temp;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA=21, SCL=22 for ESP32

  if (imu.begin() != 0) {
    Serial.println("Failed to initialize ICG-20660L!");
    while (1);
  }
  imu.enableSensor(0b00111111); // Bits 0-5 = gyro XYZ + accel XYZ

  Serial.println("ICG-20660L initialized.");

}

void loop() {
  imu.getSensorData(&accel, &gyro, &temp);

  // But only use accel
  Serial.print("Accel X: "); Serial.print(accel.x, 4);
  Serial.print(" g  Y: ");   Serial.print(accel.y, 4);
  Serial.print(" g  Z: ");   Serial.println(accel.z, 4);

  Serial.print("Temp: "); Serial.println(temp, 2);

  delay(200);
}