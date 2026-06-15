#include <Arduino_LSM9DS1.h>
#include <SimpleKalmanFilter.h>

// Für Accelerometer (G-Werte: reagieren eher langsam)
SimpleKalmanFilter kf_accX(0.1, 0.1, 0.05);
SimpleKalmanFilter kf_accY(0.1, 0.1, 0.05);
SimpleKalmanFilter kf_accZ(0.1, 0.1, 0.05);

// Für Gyroskop (dps-Werte: reagieren sehr schnell bei Wischgesten, deshalb höheres q)
SimpleKalmanFilter kf_gyrX(2.0, 2.0, 0.1);
SimpleKalmanFilter kf_gyrY(2.0, 2.0, 0.1);
SimpleKalmanFilter kf_gyrZ(2.0, 2.0, 0.1);


void setup()
{
  Serial.begin(115200); // Startet die serielle Verbindung
  while (!Serial)
    ; // Wartet, bis der Serial Monitor geöffnet wird

  if (!IMU.begin())
  {
    Serial.println("Fehler bei der Initialisierung der IMU!");
    while (1)
      ;
  }
}

void loop()
{
  float accX, accY, accZ;
  float gyrX, gyrY, gyrZ;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
  {
    IMU.readAcceleration(accX, accY, accZ); // Liest Beschleunigung in G (Erdbeschleunigung)
    IMU.readGyroscope(gyrX, gyrY, gyrZ);

    float fAccX = kf_accX.updateEstimate(accX);
    float fAccY = kf_accY.updateEstimate(accY);
    float fAccZ = kf_accZ.updateEstimate(accZ);
    
    float fGyrX = kf_gyrX.updateEstimate(gyrX);
    float fGyrY = kf_gyrY.updateEstimate(gyrY);
    float fGyrZ = kf_gyrZ.updateEstimate(gyrZ);

    // Daten im Format "X,Y,Z" ausgeben
    // Serial.print(fAccX);
    // Serial.print(",");
    // Serial.println(accX);
    // Serial.print(",");
    // Serial.print(fAccY);
    // Serial.print(",");
    // Serial.print(accY);
    // Serial.print(",");
    Serial.print(fAccZ);
    Serial.print(",");
    Serial.println(accZ);  
  }

  delay(10); // Kurze Pause, um die Datenrate anzupassen
}
