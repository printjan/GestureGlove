# Data Fusion Project


## Team

- Lucas Horn: `hornlu95907@th-nuernberg.de`
- Jan Tichner: `tischnerja95752@th-nuernberg.de`


---


## Hardware Setup

### Sensor board:

- We are using two `XIAOML Kit` devices: Designed by Professor Vijay Janapa Reddi (Harvard University), author of the Machine Learning Systems textbook. One at the wrist and one at the index finger.
- What's inside: XIAO ESP32-S3 Sense, CAM • IMU • SD Toolkit
- Advertising: Build keyword detection, image classification, motion detection, object detection, and more
- Links: For ...
  - Learners: mlsysbook.ai
  - Builders: mlsysbook.ai/kits
  - Developers: github.com/mlsysbook

### Setup:

- The two XIAOML Kits are directly conncted to the computer via USB-C.
- IMU Data will be streamed unprocessed via USB-C-Serial to the computer.
- All processing, fusion, filtering, and ML will run on the Computer. 


---


## Project description

**Setup:**
- One XIAOML Kit on the wrist (IMU Data).
- One XIAOML Kit on the tip of the index finger (Camera Data).
- Orientation usb-c-plug downward and backward.
- Mounted on right hand.
  
**Goal:**
- Recognize arm- and hand-gestures with wrist worn IMU Sensor.
- Demonstation: Control power point with hand gestures.

**Possible Extension:**
- Use finger as an air mouse to interact with the computer.
- Demonstration: Cotrol the power point laser pointer by hand movement.


---


## Guestures

**Very important:** Discrete Movement (Recognizable Start and Stop of the movement with a stationary moment before and after to differentiate the geusture from natural movement)!

### Arm gestures

**Swipe Right / Swipe Left:**
- Movement: Horizonal movement of the hand with dedicated still moment at the end and the beginning of the gesture.
- Demonstration: Next / Previous slide in powerpoint.
**Jerk Up / Jerk Down:**
- Movement: Vertical movement of the hand with dedicated still moment at the end and the beginning of the gesture.
- Demonstration: Volume Up / Volume Down.
**Circle Clockwise / Circle Counter Clockwise:**
- Movement: Clockwise / Counter Clockwise movement of the wrist with dedicated still moment at the end and the beginning of the gesture.
- Demonstration: Toggle Laser Pointer Mode.

### Hand gestures

**Make fist:**
- Movement: Close hand (make fist) and immediately open it again twice. Hand celarly open at the end and beginning of the gesture. During the gesture arm stays still
- Demonstration: Toggle Laser Pointer Mode.

### None

**None class:**
- Movement: Idle: Hold still or slightly move indiscriminateley.
- Demonstration: Speaking and moving naturally.

### Naming scheme

In the dataset an classifiers the naming scheme will be as follows:

```
[
  "none",
  "swipe_left",
  "swipe_right",
  "circle_cw",
  "circle_ccw",
  "fist",
  "jerk_down",
  "jerk_up"
]
```


---


## Project Strucure 

```
data_fusion_project/
├── data/
│   ├── <guesture name>/
│   │   │   ├── <recording_session>/
│   │   │   │   ├── calibration.csv # 5 second recording of no movement to establish sensor drift
│   │   │   │   ├── 00001.csv # first recording of the gesture
│   │   │   │   ├── 00002.csv # second recording of the gesture
│   │   │   │   └── 
```


---


## Data set structure

Column structure of `<timestamp>.csv` files:

```csv
IMU1_accX,IMU1_accY,IMU1_accZ,IMU1_gyrX,IMU1_gyrY,IMU1_gyrZ,IMU2_accX,IMU2_accY,IMU2_accZ,IMU2_gyrX,IMU2_gyrY,IMU2_gyrZ
```


---


## Pitch

- 5 min (auf keinen Fall mehr).
- 4 Folien.
  - 1. Team und Problem.
  - 2./3. Folie Implementierungsdetails.
  - 4. Folie: Gefilmte Demo (keine Live Demo).
- Schöne Animationen sind wichtig!
- Code in Git Repo der Fak Inf ablegen mit Axenie als Maintainer.
- Folien in Repo ablegen (als .pdf).
- Alle Medien (also auch Präsi Videos) auf Git ablegen.
- Mündlich darauf vorbereiten, Fragen zum Projekt zu beantworten (auch kritische).
- Abgabe: 1. Juli 23:59. Kein Commit mehr danach.