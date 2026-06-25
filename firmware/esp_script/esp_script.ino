#include <LSM6DS3.h>
#include <Wire.h>

// Create IMU object using I2C interface
LSM6DS3 myIMU(I2C_MODE, 0x6A);

float accX, accY, accZ;
float gyrX, gyrY, gyrZ;

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    delay(10);

  // Initialize the IMU
  if (myIMU.begin() != 0)
  {
    Serial.println("ERROR: IMU initialization failed!");
    while (1)
      delay(1000);
  }
}

void loop()
{
  unsigned long timestamp = micros();

  // Read accelerometer data (in g-force)
  accX = myIMU.readFloatAccelX();
  accY = myIMU.readFloatAccelY();
  accZ = myIMU.readFloatAccelZ();

  // Read gyroscope data (in degrees per second)
  gyrX = myIMU.readFloatGyroX();
  gyrY = myIMU.readFloatGyroY();
  gyrZ = myIMU.readFloatGyroZ();

  // Print readable format
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(accX);
  Serial.print(",");
  Serial.print(accY);
  Serial.print(",");
  Serial.print(accZ);
  Serial.print(",");
  Serial.print(gyrX);
  Serial.print(",");
  Serial.print(gyrY);
  Serial.print(",");
  Serial.println(gyrZ);

  delay(5); // 100 Hz update rate
}
