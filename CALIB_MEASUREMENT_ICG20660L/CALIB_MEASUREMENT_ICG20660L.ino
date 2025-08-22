#include <Arduino.h>
#include "Wire.h"
#include "DFRobot_ICG20660L.h"

constexpr float g = 9.81735f; //exact g, may be changed
constexpr int increment_angle = 15;
constexpr int measurement_amount = 300; //measurements in one position
constexpr int pos = 360 / increment_angle; //24 different positions on each axis
constexpr int all_pos = pos*3; //all of the positions on all axes combined

//the following info is based on SER0067 servo, which is used in original testbench
constexpr int servo_pin = 18; //has to be one of the PWM pins on ESP32
constexpr int freq = 50;     //50hz servo
constexpr int channel = 0;   //PWM channel, ESP32 has 16 different channels for controlling multiple PWM outputs
constexpr int resolution = 16;  //16-bit PWM resolution (max res for max precision)
const int min_us = 500;        //min pulse for 0°
const int max_us = 2500;       //max pulse for 360°
const int error_offset_us = 13;// offset from real pulse value found experimentally - this has to be subtracted from pulse value sent to servo to make it more accurate

//make sure following pins are connected on imu
const int SDA_pin = 21; //pin on ESP32
const int SCL_pin = 22;

float ax_g, ay_g, az_g; //global variables for recording values over and over
float gx_dps, gy_dps, gz_dps;


struct data_entry {
  float x, y, z;
  char current_axis;
};

void log_measurement(struct data_entry avg_measurement[], int i){
  Serial.print("&X");   //& tähistus voimaldab pythoni koodil pärast vajalikku infot eraldi faili salvestada
  Serial.print(avg_measurement[i].x, 4);  //4 komakohta
  Serial.print("Y");
  Serial.print(avg_measurement[i].y, 4);
  Serial.print("Z");
  Serial.print(avg_measurement[i].z, 4);
  Serial.print("A");
  Serial.print(avg_measurement[i].current_axis);
  Serial.println();
}

void collect_manual() {
  struct data_entry avg_measurement[all_pos];

  for (int i = 0; i < all_pos; i++) {
    int group = i / pos; // grupeerimise abil saab anda teada serial monitoris, millisest teljes jutt käib
    int true_pos = i % pos;
    
    if (group == 0) avg_measurement[i].current_axis = 'X';
    else if (group == 1) avg_measurement[i].current_axis = 'Y';
    else avg_measurement[i].current_axis = 'Z';

    Serial.println("Position ");
    Serial.print(true_pos);
    Serial.print(" - Rotation around ");
    if (group == 0) Serial.println("X-axis");
    else if (group == 1) Serial.println("Y-axis");
    else Serial.println("Z-axis");
    
    if (true_pos == 0 && group != 0){
      Serial.println("IMPORTANT: NEW AXIS! CONFIRM AXIS!");
      Serial.println("Type 'c' and press ENTER to start calibration on this axis.");
      while (true) {
        char input = get_user_input();
        if (input == 'c' || input == 'C') break;
        else Serial.println(">");
      }
    }

    Serial.println();
    Serial.println("Recording data... Please keep sensor still.");
    delay(2000); // anduril lastakse natuke seista, et vältida vibratsioone jms, mis tulevad kohe pärast käskluse saatmist kasutaja klaviatuurilt

    float sum_x = 0, sum_y = 0, sum_z = 0;

    for (int j = 0; j < measurement_amount; j++) {
      float a_x, a_y, a_z;
      read_MPU6050();

      sum_x += a_x;
      sum_y += a_y;
      sum_z += a_z;

      delay(10); // sekundis 100 mõõtmist
    }

    avg_measurement[i].x = sum_x / measurement_amount;  //võetakse aritmeetiline keskmine kõikidest mõõtmistest ühes positsioonis, et vähendada maatriksite suurust ja minimaliseerida müra
    avg_measurement[i].y = sum_y / measurement_amount;
    avg_measurement[i].z = sum_z / measurement_amount;
   
    Serial.print("Average X: "); Serial.print(avg_measurement[i].x, 4);
    Serial.print(" Y: "); Serial.print(avg_measurement[i].y, 4);
    Serial.print(" Z: "); Serial.println(avg_measurement[i].z, 4);

    if (i != (all_pos - 1)){    
      Serial.println("File log:");
      Serial.println("--------------------------------------------------");
      log_measurement(avg_measurement, i);
      Serial.println("--------------------------------------------------");

      if (true_pos == 24) Serial.println("CHANGE AXIS BEFORE NEXT CALIBRATION!"); // meeldetuletus

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

void collect_servo() {
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
      delay(2000); // waiting time to ensure that measuring platform settles

      if (axis == 0) avg_measurement[i].current_axis = 'X';
      else if (axis == 1) avg_measurement[i].current_axis = 'Y';
      else avg_measurement[i].current_axis = 'Z';

      float sum_x = 0, sum_y = 0, sum_z = 0;

      for (int j = 0; j < measurement_amount; j++) { //starts measuring in this position
        read_ICG20660L();

        sum_x += ax_g;
        sum_y += ay_g;
        sum_z += az_g;

        delay(10); // 100 measurements per second
      }

      avg_measurement[i].x = sum_x / measurement_amount;  //average is taken to minimize noise influence on calib results
      avg_measurement[i].y = sum_y / measurement_amount;
      avg_measurement[i].z = sum_z / measurement_amount;

      log_measurement(avg_measurement, i);
      Serial.println("--------------------------------------------------");
    }
    if (axis != 2) Serial.print("Change axis! Next axis: ");
    else Serial.println("Calibration completed succesfully.");

    if (axis == 0) Serial.println("Y- axis, blue marking.");
    if (axis == 1) Serial.println("Z- axis, green marking.");
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
  imu.read_sensor_data(ax_g, ay_g, az_g);

  Serial.print("Accel [g] X="); Serial.print(ax);
  Serial.print(" Y="); Serial.print(ay);
  Serial.print(" Z="); Serial.println(az);

  Serial.print("Temp [°C]="); Serial.println(temp, 2);
  delay(200);
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

  ledcAttach(servo_pin, freq, resolution);

  if (imu.begin() != 0) {
    Serial.println("Failed to initialize ICG-20660L!");
    while (1);
  }
  Serial.println("ICG-20660L initialized.");

  imu.enableSensor(ACCEL_ENABLE);  // Turn on accelerometer used for this test
  }
}

void loop() {
  Serial.println("--------------------------------------------------");
  Serial.println("Choose calibration:");
  Serial.println("1 - Manual accelerometer");
  Serial.println("2 - Servo-assisted accelerometer");
  Serial.println(">");

  char choice = get_user_input();

  if (choice == '1') {
    Serial.println("&MODE=M_ACCEL");
    collect_manual();
  }
  else if (choice == '2') {
    Serial.println("&MODE=S_ACCEL");
    collect_servo();
  }
  else {
    Serial.println("Invalid choice.");
  }

}
