#include <cmath>
#include <iostream>
#include <Adafruit_MPU6050.h>

constexpr float g = 9.81735f;
constexpr int increment = 15;
constexpr int measurement_amount = 1000; //Ühes positsioonis tehtavate mõõtmiste arv
constexpr int pos = 360 / increment; //15 kraadised inkremendid, 24 erinevat positsiooni
constexpr int all_pos = pos*3; //muutuja, mis kirjeldab kõiki positsioone, milles andur viibib kalibreerimise ajal (keeratake 3 teljes)

float T[all_pos][4];  //loome ühe suure maatriksi tõeste väärtuste hoidmiseks

void gen_T(float T[all_pos][4]) {
  for (int i = 0; i < pos; ++i) {
    float angle_deg = i * increment;
    float angle_rad = angle_deg * M_PI / 180.0f;

    // Pöörlemine ümber IMU x telje (ridadel 0 kuni pos-1) -> VALGE
    T[i][0] = 0.0f;                    // g projektsioon x teljele
    T[i][1] = g * sin(angle_rad);      // g projektsioon y teljele / märk eeldab, et y näitab SINU poole - ASETA IMU ÕIGET PIDI
    T[i][2] = -g * cos(angle_rad);     // g projektsioon z teljele / märk eeldab, et z näitab ÜLES
    T[i][3] = 1.0f;                    // neljas tulp maatriksis täidetud 1-ga, sest vajalik arvutuses

    // Pöörlemine ümber IMU y telje (ridadel pos kuni 2*pos-1) -> sensorit peab plaadi peal pöörama või ROHELINE
    T[i + pos][0] = g * sin(angle_rad); // märk eeldab, et x näitab PAREMALE
    T[i + pos][1] = 0.0f;
    T[i + pos][2] = -g * cos(angle_rad);
    T[i + pos][3] = 1.0f;

    // Pöörlemine ümber IMU z telje teisel küljel (ridadel 2*pos kuni 3*pos-1) - ROHELINE
    T[i + 2 * pos][0] = g * cos(angle_rad);
    T[i + 2 * pos][1] = -g * sin(angle_rad);
    T[i + 2 * pos][2] = 0.0f;
    T[i + 2 * pos][3] = 1.0f;
  }
}

void transpose_4xN(float source_matrix[all_pos][4], float transposed_matrix[4][all_pos]){
  for (int i=0; i<all_pos; i++){
    for (int j=0; j<4; j++){
      transposed_matrix[j][i] = source_matrix[i][j];
    }
  }
}

void multiply_4xN_Nx4(float mat_A[4][all_pos], float mat_B[all_pos][4], float result_matrix[4][4]){
  for (int i=0; i<4; i++){    //võtab esimese maatriksi read
    for (int j=0; j<4; j++){  //võtab teise maatriksi veerud
      result_matrix[i][j] = 0.0f;
      for (int k=0; k < all_pos; k++){
        result_matrix[i][j] += mat_A[i][k] * mat_B[k][j]; //loob 4x4 maatriksi arvud liites ükshaaval 4xN maatriksi rea ja Nx4 maatriksi veeru liikmete omavahelised korrutised
      }
    }
  }
}

void multiply_4xN_Nx1(float mat_A[4][all_pos], float mat_B[all_pos], float result_matrix[4]){
  for (int i=0; i<4; i++){
    result_matrix[i] = 0.0f;;
    for (int k=0; k < all_pos; k++){
      result_matrix[i] += mat_A[i][k] * mat_B[k];
    }
  }
}

bool invert_4x4(const float m[4][4], float inv_out[4][4]) 
//see on täielikult stackoverflowst võetud ja 2D maatriksi kujule üle viidud inversioonitehnika
{
    float inv[4][4], det; 

    inv[0][0] = m[1][1]  * m[2][2] * m[3][3] - 
             m[1][1]  * m[3][2] * m[2][3] - 
             m[1][2]  * m[2][1]  * m[3][3] + 
             m[1][2]  * m[3][1]  * m[2][3] +
             m[1][3] * m[2][1]  * m[3][2] - 
             m[1][3] * m[3][1]  * m[2][2];

    inv[0][1] = -m[0][1]  * m[2][2] * m[3][3] + 
              m[0][1]  * m[3][2] * m[2][3] + 
              m[0][2]  * m[2][1]  * m[3][3] - 
              m[0][2]  * m[3][1]  * m[2][3] - 
              m[0][3] * m[2][1]  * m[3][2] + 
              m[0][3] * m[3][1]  * m[2][2];

    inv[0][2] = m[0][1]  * m[1][2] * m[3][3] - 
             m[0][1]  * m[3][2] * m[1][3] - 
             m[0][2]  * m[1][1] * m[3][3] + 
             m[0][2]  * m[3][1] * m[1][3] + 
             m[0][3] * m[1][1] * m[3][2] - 
             m[0][3] * m[3][1] * m[1][2];

    inv[0][3] = -m[0][1]  * m[1][2] * m[2][3] + 
               m[0][1]  * m[2][2] * m[1][3] +
               m[0][2]  * m[1][1] * m[2][3] - 
               m[0][2]  * m[2][1] * m[1][3] - 
               m[0][3] * m[1][1] * m[2][2] + 
               m[0][3] * m[2][1] * m[1][2];

    inv[1][0] = -m[1][0]  * m[2][2] * m[3][3] + 
              m[1][0]  * m[3][2] * m[2][3] + 
              m[1][2]  * m[2][0] * m[3][3] - 
              m[1][2]  * m[3][0] * m[2][3] - 
              m[1][3] * m[2][0] * m[3][2] + 
              m[1][3] * m[3][0] * m[2][2];

    inv[1][1] = m[0][0]  * m[2][2] * m[3][3] - 
             m[0][0]  * m[3][2] * m[2][3] - 
             m[0][2]  * m[2][0] * m[3][3] + 
             m[0][2]  * m[3][0] * m[2][3] + 
             m[0][3] * m[2][0] * m[3][2] - 
             m[0][3] * m[3][0] * m[2][2];

    inv[1][2] = -m[0][0]  * m[1][2] * m[3][3] + 
              m[0][0]  * m[3][2] * m[1][3] + 
              m[0][2]  * m[1][0] * m[3][3] - 
              m[0][2]  * m[3][0] * m[1][3] - 
              m[0][3] * m[1][0] * m[3][2] + 
              m[0][3] * m[3][0] * m[1][2];

    inv[1][3] = m[0][0]  * m[1][2] * m[2][3] - 
              m[0][0]  * m[2][2] * m[1][3] - 
              m[0][2]  * m[1][0] * m[2][3] + 
              m[0][2]  * m[2][0] * m[1][3] + 
              m[0][3] * m[1][0] * m[2][2] - 
              m[0][3] * m[2][0] * m[1][2];

    inv[2][0] = m[1][0]  * m[2][1] * m[3][3] - 
             m[1][0]  * m[3][1] * m[2][3] - 
             m[1][1]  * m[2][0] * m[3][3] + 
             m[1][1]  * m[3][0] * m[2][3] + 
             m[1][3] * m[2][0] * m[3][1] - 
             m[1][3] * m[3][0] * m[2][1];

    inv[2][1] = -m[0][0]  * m[2][1] * m[3][3] + 
              m[0][0]  * m[3][1] * m[2][3] + 
              m[0][1]  * m[2][0] * m[3][3] - 
              m[0][1]  * m[3][0] * m[2][3] - 
              m[0][3] * m[2][0] * m[3][1] + 
              m[0][3] * m[3][0] * m[2][1];

    inv[2][2] = m[0][0]  * m[1][1] * m[3][3] - 
              m[0][0]  * m[3][1] * m[1][3] - 
              m[0][1]  * m[1][0] * m[3][3] + 
              m[0][1]  * m[3][0] * m[1][3] + 
              m[0][3] * m[1][0] * m[3][1] - 
              m[0][3] * m[3][0] * m[1][1];

    inv[2][3] = -m[0][0]  * m[1][1] * m[2][3] + 
               m[0][0]  * m[2][1] * m[1][3] + 
               m[0][1]  * m[1][0] * m[2][3] - 
               m[0][1]  * m[2][0] * m[1][3] - 
               m[0][3] * m[1][0] * m[2][1] + 
               m[0][3] * m[2][0] * m[1][1];

    inv[3][0] = -m[1][0] * m[2][1] * m[3][2] + 
              m[1][0] * m[3][1] * m[2][2] + 
              m[1][1] * m[2][0] * m[3][2] - 
              m[1][1] * m[3][0] * m[2][2] - 
              m[1][2] * m[2][0] * m[3][1] + 
              m[1][2] * m[3][0] * m[2][1];

    inv[3][1] = m[0][0] * m[2][1] * m[3][2] - 
             m[0][0] * m[3][1] * m[2][2] - 
             m[0][1] * m[2][0] * m[3][2] + 
             m[0][1] * m[3][0] * m[2][2] + 
             m[0][2] * m[2][0] * m[3][1] - 
             m[0][2] * m[3][0] * m[2][1];

    inv[3][2] = -m[0][0] * m[1][1] * m[3][2] + 
               m[0][0] * m[3][1] * m[1][2] + 
               m[0][1] * m[1][0] * m[3][2] - 
               m[0][1] * m[3][0] * m[1][2] - 
               m[0][2] * m[1][0] * m[3][1] + 
               m[0][2] * m[3][0] * m[1][1];

    inv[3][3] = m[0][0] * m[1][1] * m[2][2] - 
              m[0][0] * m[2][1] * m[1][2] - 
              m[0][1] * m[1][0] * m[2][2] + 
              m[0][1] * m[2][0] * m[1][2] + 
              m[0][2] * m[1][0] * m[2][1] - 
              m[0][2] * m[2][0] * m[1][1];

    det = m[0][0] * inv[0][0] + m[1][0] * inv[0][1] + m[2][0] * inv[0][2] + m[3][0] * inv[0][3];

    if (det == 0)
        return false;

    det = 1.0f / det;

    for (int i = 0; i < 4; i++){
      for(int j = 0; j < 4; j++){
        inv_out[i][j] = inv[i][j] * det;
      }
    }

    return true;
}

void calibrate_axis(float T[][4], float M[], float result[4]) {
  float Tt[4][all_pos];         // transpose of T
  float TtT[4][4];              // T^T * T
  float TtM[4];                 // T^T * M
  float invTtT[4][4];           // inverse of T^T * T

  transpose_4xN(T, Tt);
  multiply_4xN_Nx4(Tt, T, TtT);
  multiply_4xN_Nx1(Tt, M, TtM);

  if (!invert_4x4(TtT, invTtT)) {
    Serial.println("Matrix inversion failed. Likely a faulty calibration process.");
    return;
  }

  // Result x = inv(T^T * T) * T^T * M
  for (int i = 0; i < 4; i++) {
    result[i] = 0.0f;
    for (int j = 0; j < 4; j++) {
      result[i] += invTtT[i][j] * TtM[j];
    }
  }
}

void read_MPU6025(float* ax, float* ay, float* az) {
  int16_t ax_raw, ay_raw, az_raw;
  imu.getAcceleration(&ax_raw, &ay_raw, &az_raw);

  *ax = (float)(ax_raw / 16384.0f) * g;
  *ay = (float)(ay_raw / 16384.0f) * g;
  *az = (float)(az_raw / 16384.0f) * g;
}

void calibrate_manual() {

  float a_x_avg[all_pos], a_y_avg[all_pos], a_z_avg[all_pos];

  for (int i = 0; i < pos * 3; i++) {
    int group = i / pos; // grupeerimise abil saab anda teada serial monitoris, millisest teljes jutt käib
    int true_pos = i % pos;
    
    Serial.print("Position ");
    Serial.print(true_pos);
    Serial.print(" - Rotation around ");
    if (group == 0) Serial.println("X-axis");
    else if (group == 1) Serial.println("Y-axis");
    else Serial.println("Z-axis");

    Serial.println("Place sensor and hold steady.");
    Serial.println("Type 'c' and press ENTER to start calibration at this position.");
    if (true_pos == 1) Serial.println("IMPORTANT: NEW AXIS! CONFIRM AXIS!");

    while (true) {
      if (Serial.available()) {
        char ch = Serial.read();
        if (ch == 'c' || ch == 'C') break;
      }
    }

    Serial.println("Recording data... Please keep sensor still.");
    delay(2000); // anduril lastakse natuke seista, et vältida vibratsioone jms, mis tulevad kohe pärast käskluse saatmist kasutaja klaviatuurilt

    float sum_x = 0, sum_y = 0, sum_z = 0;

    for (int j = 0; j < measurement_amount; j++) {

      imu.getAcceleration(&ax_raw, &ay_raw, &az_raw); // käsklus IMU-lt andmete saamiseks (oleneb IMU-st)

      float a_x = (float)ax_raw / 16384.0f * g; //read 97-99 on vajalikud ainult siis kui andur annab andmeid toorelt 16384 ühiku skaalal (see on 1g)
      float a_y = (float)ay_raw / 16384.0f * g;
      float a_z = (float)az_raw / 16384.0f * g;

      sum_x += a_x;
      sum_y += a_y;
      sum_z += a_z;

      delay(10); // sekundis 100 mõõtmist
    }

    a_x_avg[i] = sum_x / measurement_amount;  //võetakse aritmeetiline keskmine kõikidest mõõtmistest ühes positsioonis, et vähendada maatriksite suurust ja minimaliseerida müra
    a_y_avg[i] = sum_y / measurement_amount;
    a_z_avg[i] = sum_z / measurement_amount;

    Serial.print("Average X: "); Serial.print(a_x_avg[i], 4);
    Serial.print(" Y: "); Serial.print(a_y_avg[i], 4);
    Serial.print(" Z: "); Serial.println(a_z_avg[i], 4);

    if (true_pos == 24) Serial.println("CHANGE AXIS BETWEEN NEXT CALIBRATION!");
    else Serial.println("Move to next position and rotate around the same axis.");

    Serial.println("Press ENTER to continue...");

    while (!Serial.available()) { }
    while (Serial.available()) Serial.read(); // clear input
  }

  Serial.println("Calculating calibration parameters...");
  float Kx[4], Ky[4], Kz[4];
  calibrate_axis(T, a_x_avg, Kx);
  calibrate_axis(T, a_y_avg, Ky);
  calibrate_axis(T, a_z_avg, Kz);

}

void collect_servo(){}

void calibrate_gyro(){

}

void setup() {
  Serial.begin(115200);
  gen_T(T);
  while(true){
    Serial.println("Choose calibration:");
    Serial.println("1 - Manual accelerometer");
    Serial.println("2 - Servo-assisted accelerometer");
    Serial.println("3 - Servo-assisted position gyroscope");
    
    while (!Serial.available()) {
      delay(10); // wait for user input
    }
    
    char choice = Serial.read();
    while (Serial.available()) {
      Serial.read();
    }

    if (choice == '1') {
        calibrate_manual();
        break;
      }
    else if (choice == '2') {
        collect_servo();
        break;
      }
    else if (choice == '3') {
        calibrate_gyro();
        break;
      }
    else {
      Serial.println("Invalid choice, restarting...");
      // 
    }
  }
}

void loop() {

}
