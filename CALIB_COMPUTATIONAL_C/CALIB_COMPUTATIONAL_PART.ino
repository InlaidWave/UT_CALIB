#include <cmath>
#include <iostream>

constexpr float g = 9.81735f;
constexpr int increment = 15;
constexpr int pos = 360 / increment; //15 kraadised inkremendid, 24 erinevat positsiooni
constexpr int measurement_amount = 1000; //Ühes positsioonis tehtavate mõõtmiste arv
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

void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:

}
