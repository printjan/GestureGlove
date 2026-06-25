#include <Arduino.h>
#include <LSM6DS3.h>
#include <Wire.h>

LSM6DS3 myIMU(I2C_MODE, 0x6A);

static const unsigned long SAMPLE_PERIOD_US = 10000; // 100 Hz
unsigned long next_sample_us = 0;

void setup() {
  Serial.begin(115200);

  // Nicht endlos blockieren, falls kein Serial Monitor offen ist.
  unsigned long start_ms = millis();
  while (!Serial && millis() - start_ms < 3000) {
    delay(10);
    git merge-- no - ff-- no - commit origin / main
  }

  Wire.begin();

  if (myIMU.begin() != 0) {
    Serial.println("ERROR: IMU initialization failed!");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("IMU_STREAM_READY");
  next_sample_us = micros();
}

void loop() {
  unsigned long now = micros();

  if ((long)(now - next_sample_us) < 0) {
    return;
  }

  next_sample_us += SAMPLE_PERIOD_US;

  unsigned long timestamp = micros();

  float accX = myIMU.readFloatAccelX();
  float accY = myIMU.readFloatAccelY();
  float accZ = myIMU.readFloatAccelZ();

  float gyrX = myIMU.readFloatGyroX();
  float gyrY = myIMU.readFloatGyroY();
  float gyrZ = myIMU.readFloatGyroZ();

  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(accX, 6);
  Serial.print(",");
  Serial.print(accY, 6);
  Serial.print(",");
  Serial.print(accZ, 6);
  Serial.print(",");
  Serial.print(gyrX, 6);
  Serial.print(",");
  Serial.print(gyrY, 6);
  Serial.print(",");
  Serial.println(gyrZ, 6);
}
