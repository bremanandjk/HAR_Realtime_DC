IMU-Based Gesture Recognition Project:

This project implements a low-resource, single IMU-based system for gesture recognition and real-time human-computer interaction. It includes data collection, model training, real-time classification, and performance visualization.

Software Requirements:
1. Programming Environment

Visual Studio Code (VS Code): Integrated development environment for managing and executing Python scripts.

Python 3.9+: Core programming language for system implementation.

2. Libraries and Dependencies

The following Python libraries are required:

bleak: For Bluetooth Low Energy (BLE) communication with the IMU device

pandas, numpy: For data manipulation and numerical computations

Project Scripts:
1. Code_Runtime_logger.py

Reads sensor data from the IMU device connected via Bluetooth

Stores the collected readings in CSV format for further processing

2. Code_Activity_Classifier_RF.py

Builds and trains a Random Forest classifier using the collected dataset

Evaluates model performance on gesture classification

3. Code_Activity_Classifier_XGboost.py

Builds and trains an XGBoost classifier using the collected dataset

Compares performance against the Random Forest model

4. Code_real_time_system_RF.py and Code_real_time_system_XGboost.py

Performs real-time gesture recognition using the wrist-worn IMU sensor

Detects gestures as they occur and displays results on the screen

5. Code_new_visual.py

Generates visualizations for model performance, including accuracy and F1 score

Supports analysis of learning curves and effect of training data size