# Jan Private Notes




- IMU Data


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
    - Hold still or slightly move indiscriminateley: (Demonstration: Idle.)
    - Make fist (Close and immediateley open fist again) (Demonstration: Toggle Laser Pointer Mode.)
- Possible Extensions:
  - 2. (Optional) Use arm Gestors as an air mouse to interact with the computer.
    - Make circle: clockwise / counter clockwise (Demonstration: Toggle Air Mouse Mode) 
    - Make fist: (Demonstration: Click.)
  - 3. (Optional) Also recognize Hand gestures with webcam.
- Demonstration: 
  - Control power point slides by hand gestures.
  - Cotrol the power point laser pointer by hand movement.


## Technology Stack





## Data structure

off-git data storage:
```
data/
  raw/
    session_xxx/
      wrist.csv
      index.csv
      labels.csv
      session_metadata.json
  processed/
  models/
```

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

possible gestures:
```
[
  "idle",
  "swipe_left",
  "swipe_right",
  "circle_cw",
  "circle_ccw",
  "fist",
  "down"
  "up"
]
```





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






