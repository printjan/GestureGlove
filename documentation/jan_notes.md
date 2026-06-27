# Jan Private Notes


## Hardware Setup

Sensor board:
- We are using two `XIAOML Kit` devices: Designed by Professor Vijay Janapa Reddi (Harvard University), author of the Machine Learning Systems textbook. One at the wrist and one at the index finger.
- What's inside: XIAO ESP32-S3 Sense, CAM • IMU • SD Toolkit
- Advertising: Build keyword detection, image classification, motion detection, object detection, and more
- Links: For ...
  - Learners: mlsysbook.ai
  - Builders: mlsysbook.ai/kits
  - Developers: github.com/mlsysbook
Setup:
- The two XIAOML Kits are directly conncted to the computer via USB-C.
- IMU Data will unprocessed be streamed via USB-C-Serial to the computer.
- All processing, fusion, filtering, and ML will run on the Computer. 


## Project Idea: Gesture Recognition

- Setup:
  - One XIAOML Kit on the wrist (IMU Data).
  - One XIAOML Kit on the tip of the index finger (Camera Data).
  - Orientation usb-c-plug downward and backward.
  - Mounted on right hand.
- Goal:
  - 1. Recognize Arm-Gestures:
    - Swipe: right / left (Demonstration: Next / Previous slide in powerpoint.)
    - Short upward / downward jerk (Demonstration: Volume Up / Volume Down.) 
    - Wrist circle clockwise / counter clockwise (Demonstration: Toggle Laser Pointer Mode.)
    - Idle (Dedicated `None` class) (Hold still or slightly move indiscriminateley): (Demonstration: Idle.)
    - Make fist (Close and immediateley open fist again) (Demonstration: Toggle Laser Pointer Mode.)
- Possible Extensions:
  - 2. (Optional) Use arm Gestors as an air mouse to interact with the computer.
    - Make circle: clockwise / counter clockwise (Demonstration: Toggle Air Mouse Mode) 
    - Make fist: (Demonstration: Click.)
  - 3. (Optional) Also recognize Hand gestures with webcam.
- Demonstration: 
  - Control power point slides by hand gestures.
  - Cotrol the power point laser pointer by hand movement.


---


## Gestures

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
- Movement: Close hand (make fist) and immediately open it again twice. Hand celarly open at the end and beginning of the gesture. During the gesture hand and arm stay still
- Demonstration: Toggle Laser Pointer Mode.

### Idle

**None class:**
- Movement: Hold still or slightly move indiscriminateley.
- Demonstration: Speaking and moving naturally.



---



## Dataset structure







## Technology Stack





## Data structure


session_metadata.json:
```
{
  "participant_id": "p01",
  "session_id": "s01",
  "date": "...",
  "sampling_rate_hz": 100,
  "gestures": []
}
```

data/





# AI Agent Notes


## Assistant Prompt:

Context:
- This is about the Data Fusion module in the sixth semester of the computer science bachelor's program. 
- The task in question is a project based on the lecture content. 
- We are free to use an existing dataset or collect our own data. 

Hardware:
- We are using `the XIAOML Kit`
  - A hands-on introduction to machine learning systems using TinyML. Designed by Professor Vijay Janapa Reddi (Harvard University), author of the Machine Learning Systems textbook.
- What's inside: XIAO ESP32-S3 Sense, CAM • IMU • SD Toolkit
- For ...
  - Learners: mlsysbook.ai
  - Builders: mlsysbook.ai/kits
  - Developers: github.com/mlsysbook

Project Idea: Gesture Recognition (right arm)
- Goal: 
  - Recognize Arm-Gestures with wrist worn IMU Sensor.
  - Use arm Gestors as an air mouse to interact with the computer.
- Possible Extension: 
  - Also recognize Hand gestures with webcam.
- Demonstration: 
  - Control power point slides by hand gestures.
  - Cotrol the power point laser pointer by hand movement.
Important Limitation: Our goal is not to build a product that works reliably in every situation with every hand. Our goal is to build a proof of concept, that works for our presentation with my hand.


## Process


Currently the following is our plan to proceed with the project:
- Install software and setup hardware.
- Detail Project Goals and Description (Which Gestures are realistic? What framework?).
- Develop Concept for Data Fusion and Data Input.
- Develop Concept for real time Classification (Which Models do we need? Which Training Data do we need?).
- Develop Concept for model training (How can we train?).
- Develop Concept for Training-Data Collection.

Analyse the project. Then Look at our persumed process - Build a Detailed Description of what we need to do and what tools we can use etc. for each step!



---


## Features

pitch, roll, yaw, acc_mag, gyro_mag, sqrt(ax²+ay²+az²), sqrt(gx²+gy²+gz²), |a|, |g|, d(ax)/dt, d(ay)/dt, d(az)/dt



---



# Python Environment


On my macbook I used the following commands to set up a virtual python environment for this project using miniconda:
```
conda config --prepend envs_dirs /opt/homebrew/Caskroom/miniconda/base/envs 
conda create -n data_fusion_env_1 python=3.11 -y
conda activate data_fusion_env_1
python -m pip install --upgrade pip setuptools wheel

conda install -c conda-forge -y \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  matplotlib \
  pyserial \
  tqdm \
  joblib \
  ipykernel \
  jupyterlab \
  notebook \
  pyyaml \
  h5py \
  pyarrow \
  filterpy \
  tensorboard

python -m pip install torch torchvision torchaudio

python -m pip install \
  opencv-python \
  mediapipe \
  pyautogui \
  pynput

python -m pip install tensorflow tensorflow-metal

python -m ipykernel install --user \
  --name data_fusion_env_1 \
  --display-name "Python (data_fusion_env_1)"
```
Write a Jupyter Notebook Script, which I can run in the project folder and that creates a matching pyhton environment `data_fusion_env_1` on my windows 11 computer with a nvidia 3080 gpu and a ryzen 9 cpu in `c:\ProgramData\python_envs`.

Also explain how I can figure out on my windows 11 machine which XIAOML Kit is connected to which COM Port.



```
conda activate data_fusion_env_1
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



---



# Current Issues


---



## Recording Pipeline



--- 



## CNN Implementation




---


## Real time inference


### Real time calibration

Strategy idea: Because the drift is dynamic and non-linear, a single initial calibration is insufficient. To keep the demo seamless without forcing the user to pause manually, we should implement Zero-Velocity Updates (ZUPT) in the background: when the hand is resting (std dev is very low for a few seconds), the system should automatically update the gyroscope bias registers in real-time. How it works: The system continuously monitors the standard deviation of gyroscope and accelerometer signals. When it detects a sustained still window (e.g. hand resting on the table for 2 seconds where std < 3.0 dps and < 0.025g), it automatically recalculates the mean zero-bias and updates the calibration profile registers in the background.



