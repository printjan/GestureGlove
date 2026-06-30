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



```
*   `IMU1_linear_jerkX`
*   `IMU1_linear_jerkZ`
*   `IMU2_linear_jerkZ`
*   `IMU1_angular_accelerationY`
*   `IMU1_angular_accelerationZ`
*   `IMU2_angular_accelerationY`
*   `IMU1_accX`
*   `IMU1_accZ`
*   `IMU1_gyrX`
*   `IMU1_pitch
*   `IMU2_accX`
*   `IMU2_accY`
*   `IMU2_accZ`
*   `IMU2_gyrX`
*   `diff_accX`
*   `diff_accZ`
*   `IMU1_gyr_mag`
*   `IMU1_accY`
*   `IMU1_gyrY`
*   `IMU1_gyrZ`
*   `IMU1_acc_mag`
*   `IMU1_roll`
*   `IMU1_relative_yaw`
*   `IMU1_linear_jerkY`
*   `IMU1_angular_accelerationX`
*   `IMU2_gyrY`
*   `IMU2_gyrZ`
*   `IMU2_gyr_mag`
*   `IMU2_acc_mag`
*   `IMU2_relative_yaw`
*   `IMU2_linear_jerkX`
*   `IMU2_linear_jerkY`
*   `IMU2_angular_accelerationX`
*   `IMU2_angular_accelerationZ`
*   `diff_accY`
*   `diff_gyrX`
*   `diff_gyrY`
*   `diff_gyrZ`
```



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

pitch, roll, yaw, acc_mag, gyro_mag, sqrt(ax²+ay²+az²), sqrt(gx²+gy²+gz²), |a|, |g|, d(ax)/dt, d(ay)/dt, d(az)/dt, Kinematic Invariants, Linear Acceleration, Differential Dynamics and Relative Yaw 



---


## Model Training (Late fusion)


### Potential Issue: The Real-Time Centering Latency Mismatch (Causal vs. Non-Causal)

- **The Issue:** 
  - The current offline pipeline uses a non-causal centering algorithm: it records a $1.6\text{ s}$ window, calculates the centroid $\mu$ across the entire window, and crops the window around it (shifting the start point backwards or forwards). 
  - During real-time inference (e.g., sliding window over serial), we cannot shift a window backwards in time without introducing latency (waiting for the gesture to end before classifying it).
- **The Risk:** If the CNN is trained only on perfectly centered gestures, it will perform poorly in a real-time sliding window where the gesture is constantly moving from the right edge of the window to the left edge.
- **Possible actions:**
  - Add Jitter Augmentation: During dataset loading, apply temporal jitter (e.g., randomly shift the window start index by $\pm 10$ or $\pm 15$ samples). This forces the CNN to learn translation invariance.
  - Use a Trigger-Based Inference Loop: Instead of classifying continuously, run a low-latency threshold check on motion energy. Once energy exceeds a threshold, record for exactly $1.5\text{ s}$, run the centroid centering on that chunk, and perform a single classification.
- **Discussion:**
  1. Increase the recording window to 1,7 seconds with 0,1 hidden recording seconds before and after the guesture.
  2. Don't delete the extra recorded data - instead store the information about the 150 datapoints in the sample, that perfectly center the guesture, in an additional .txt file with the same index as the recorded sample.
  3. When calculating graphs and/or evaluating how centered the guestures in our samples are, use the <index>.txt file to cut the relevant 150 datapoints out of the sample.
  4. During a first training round only use the perfectly centered guestures by selecting the relevant 150 datapoints out of the sample using the <index>.txt files.
  5. We can analyze how the model will react if the guesture is not perfectly centered in the processed window by cutting the 150 datapoints out of our recoring windows closer to the edges.
  6. We only need our model to give a positive result for a gesture on two or three processesd windwos - that is enough to trigger the accoring event. So it is not important for us, that our model perfectly says during which period a gesture accured, but it is important, that our model clearly says if a gesture was performed and what gesture was performed. We fear, that if we train our models on data where the gestures are not centered enough in the recording samples, our modles might loose discriminatory capabilties - they might get better at predicting if there was a gesture but not what gesture it was exactly. Is that a valid concern?





Now update the plotting logic:
During data recording we currently only keep a energy_distribution_<id>.png plot of the 150 datapoints defined in the .txt files - rename that to centered_energy_distribution_<id>.png. Add a overall_energy_distribution_<id>.png plot - that plot contains the energy distribution of the whole recording and markers for where the begin end end indices defined in the .txt fall on average. 

Also update the `scripts/analyze_motion_energy.ipynb` to respect the new data structure!
Aditionally add a new chapter to the dataset analysis in the notebook comparing the movement energy when selecting the first 150 datapoints, the last 150 datapoints and the centered 150 datapoints (as desvribed by the .txt files) from the samples: overlay their energy distribution in one plot for each guesture - this allows to visually see the impact of moving the the 150-point-gesture-window in the recorded sample! To make everything better visible also mark the peaks of the three distributions and keep the overall y-axis and x-axis range similar for all distributions!

Should we keep any additional statistics about our data?



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



### Project strategy




## Analyze the data


Go over the jupyter notebook analyzing the features and filters we plan on using to train and run our models again - Check if it really matches all the requirements we specified for it and make adustments where needed!

Here are the features with their filters as specified in the `model_training.md`:
```
## Real-Time Pre-Computed Features & Filtering Requirements


### First-Order Time Derivatives (Jerk & Angular Acceleration)

* **Linear Jerk ($J_t$)**:
  $$J_t = \frac{\mathbf{a}_t - \mathbf{a}_{t-1}}{\Delta t}$$
* **Angular Acceleration ($\alpha_t$)**:
  $$\alpha_t = \frac{\mathbf{g}_t - \mathbf{g}_{t-1}}{\Delta t}$$
* **Benefit**: Jerk is highly discriminative for "snappy" vs. "smooth" gestures. For example, `jerk_up` has an extremely high transient movement-energy peak, whereas `circle_cw` has a low, continuous movement-energy profile.
* **Filtering Strategy:** Differentiation acts as a high-pass operation that multiplies high-frequency noise. Jerk **must be low-pass filtered** (either by smoothing raw accelerometer inputs at 8.0 Hz prior to differentiation, or by smoothing the computed jerk channel itself) to prevent complete noise saturation.


### Inter-IMU Differential Features (Finger vs. Wrist)

* **Relative Acceleration ($\Delta a$) & Relative Rotation ($\Delta g$)**:
  $$\Delta a = \mathbf{a}_{finger} - \mathbf{a}_{wrist}, \quad \Delta g = \mathbf{g}_{finger} - \mathbf{g}_{wrist}$$
* **Benefit**: Distinguishes **arm gestures** from **hand gestures**.
  * During a `swipe_right`, the whole arm moves as a rigid body: $\Delta a \approx 0$ and $\Delta g \approx 0$.
  * During a `fist` or finger gesture, only the finger moves relative to the wrist: $\Delta a \gg 0$.
* **Filtering Strategy:** Relies on low-pass filtered raw inputs (8.0 Hz for accelerometer, 12.0 Hz for gyroscope) to keep differentiation clean and prevent noise propagation between the two sensors.
* **Mounting Alignment Assumption:** This simple axis-wise subtraction assumes that the two IMUs (Wrist and Finger) are mounted with **parallel coordinate axes** (specifically, both devices must have their USB-C ports pointing backward and downward). If the sensors were mounted with different orientations, the subtraction would yield arbitrary vector directions. This strict physical setup constraint replaces the need for expensive real-time coordinate transformation matrix multiplications.


### Short-Term Window Integrals (Relative Yaw)

While absolute Yaw cannot be anchored without a magnetometer, relative yaw change ($\Delta \psi$) over the short gesture window ($1.5\text{ s}$) is highly accurate and drift-free.

* **Methodology**: Integrate the z-gyroscope channel over the active window:
  $$\Delta \psi = \sum_{t \in W} g_{z,t} \cdot \Delta t$$
* **Benefit**: Provides clear directional mapping in the horizontal plane (distinguishing clock rotations and horizontal sweeps).
* **Note**: Absolute Yaw accumulates continuous non-linear drift. Will corrupt decision boundaries over time without a magnetometer.
* **Filtering Strategy:** The input gyroscope signal must be zero-bias calibrated and high-pass filtered (0.5 Hz cutoff) prior to integration to remove the DC offset and prevent linear drift over the 150-sample window.


### Kinematic Invariants (Magnitudes)

* **Accelerometer Magnitude ($a_{mag}$)**:
  $$a_{mag} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
  * *Benefit*: Independent of sensor orientation. Under static conditions (no motion), $a_{mag} \approx 1.0g$. Any deviation ($|a_{mag} - 1.0| > \epsilon$) indicates linear acceleration, which is a perfect trigger for separating active gestures from `none`.
* **Gyroscope Magnitude ($g_{mag}$)**:
  $$g_{mag} = \sqrt{g_x^2 + g_y^2 + g_z^2}$$
  * *Benefit*: Captures net angular velocity. Distinguishes rotational movements (like `circle_cw`) from purely linear movements (like `jerk_down`).
* **Filtering Strategy:** Squaring rectifies and amplifies high-frequency noise peaks. Magnitude outputs should be low-pass filtered to create a smooth, clean envelope representing physical motion energy.


### Gravity-Free Linear Acceleration ($a_{linear}$)

The raw accelerometer measures both physical acceleration and the static gravity vector ($1.0g$).

* **Methodology**: Use the calculated pitch ($\theta$) and roll ($\phi$) from your sensor fusion filter (Kalman/Complementary) to project gravity out:
  $$\mathbf{a}_{linear} = \mathbf{a}_{raw} - \mathbf{g}_{rotated}$$
* **Benefit**: Distinguishes directional gestures (like `swipe_left` vs `swipe_right`) from simple posture rotations.
* **Filtering Strategy:** Raw accelerometer signals and the estimated orientation angles must be low-pass filtered to align their phase before performing gravity projection, preventing artifacts from temporal lag.
```

When we defined the features we also developed diefferent mechanisms of how we could evaluate their suitability for our project goals:
```
## Real-Time Pre-Computed Features & Selection Rationale


### A. Feature Selection Strategy: Overengineering vs. Real-Time Efficiency

* **The Theoretical Reality:** 
  * In modern deep learning, a sufficiently deep CNN can theoretically learn to extract optimal spatial-temporal representations directly from raw multi-channel IMU data ($a_x, a_y, a_z, g_x, g_y, g_z$). 
  * But pre-calculating physical invariants and kinematic features can dramatically reduce the required network complexity, speed up convergence, and increase robustness to sensor orientation deviations.
  * Gradient descent naturally suppresses useless or noisy features by decaying their weight parameters, meaning pre-training feature selection is not strictly *necessary* for the model to work.

* **The Practical Constraints of Real-Time:** In our target application (real-time PowerPoint control running continuously), feeding raw data and relying entirely on the network to figure it out has three severe drawbacks:
  1. **Computational Overhead & Latency:** Real-time inference on a host machine runs a continuous sliding window (in our case at 100 Hz). Pre-calculating complex features (like Kalman-filtered Euler angles or sliding derivatives) consumes precious floating-point CPU cycles, adding latency.
  2. **Curse of Dimensionality:** With small training sets, adding too many noisy or redundant input channels increases model parameter size, raising the risk of overfitting.
  3. **Sensor-Defect Sensitivity (Yaw Drift):** Standard IMUs (without magnetometers) suffer from continuous yaw drift. If the absolute yaw is fed to a model, the CNN will overfit to the absolute angle of the room, failing after just a few minutes of use as drift accumulates.

* **The Goal:** We want a **minimalist, highly discriminative feature set** where every channel is physically justified and computed in $O(1)$ time complexity per sample step.



### How to Analyze Feature Utility Quantitatively

To avoid subjective overengineering, we can audit candidate features before settling on the final set:

1. **Mutual Information (MI) Regression/Classification:** Calculate the Mutual Information score between each candidate feature channel and the target gesture labels. Channels with MI $\to 0$ are candidates for exclusion.
2. **Random Forest Gini Importance:** Flatten the windows and train a lightweight Random Forest. Review the feature importance weights to identify channels that do not contribute to splitting decision boundaries.
3. **Ablation Studies:** During early training runs, systematically train the CNN with and without specific feature groups (e.g. Raw + Differentials vs. Raw + Differentials + Jerk) and check if validation accuracy changes significantly.
```

- Add detailed explanations for each experiment detailing what exaclty we want to investigate with each test and how we want to investigate it with that particular test!
- Include the dynamic dataset loading mechanism provided by the data_processing pipeline!
- Fully utilize the functionality the python module data_fusion_project.core provides!

Implement these tests in your notebook and analyze the results in relation to the projects ultimate goals outlined in `README.md`. Specifically analyze the results in relation to the future steps and experiments we have defined in `model_training.md` - this will give us a better understanding of what we can expect from the different features and filters we plan on using to train and run our models and if we should adjust our strategy accordingly.  Document your analysis in a clear and concise manner, with appropriate visualizations and explanations.

Directly incoporate any neccessary adjustments into `model_training.md` as you see fit so the notebooks and the documentation are in sync!



## CNN Implementation


### Update CNN Training implementation plans

Earlier we defined three implementation plans for three different CNN architectures we plan on comparing on our dataset in `early_fusion_single_branch_1d_cnn.md`, `late_fusion_multi_branch_1d_cnn.md` and `slef_attention_temporal_transformer.md`. Use the knowledge we gathered while auditing our dataset and the features it produces to update these implementation plans. Do not forget to update `model_training.md` accordingly if neccessary! Respect the structure we have defined for storing our model training experiment in the `README.md`. Do not repeat yourself and only add or update what is neccessary. 



---



## Real time inference


### Real time calibration

Strategy idea: Because the drift is dynamic and non-linear, a single initial calibration is insufficient. To keep the demo seamless without forcing the user to pause manually, we should implement Zero-Velocity Updates (ZUPT) in the background: when the hand is resting (std dev is very low for a few seconds), the system should automatically update the gyroscope bias registers in real-time. How it works: The system continuously monitors the standard deviation of gyroscope and accelerometer signals. When it detects a sustained still window (e.g. hand resting on the table for 2 seconds where std < 3.0 dps and < 0.025g), it automatically recalculates the mean zero-bias and updates the calibration profile registers in the background.




