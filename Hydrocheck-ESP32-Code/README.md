# Hydrocheck ESP32 Firmware

This directory contains the C++/Arduino firmware for the Hydrocheck wearable device. It reads biometric and environmental sensors, runs real-time TinyML inference on-device, drives an OLED display, and transmits data over Bluetooth Low Energy (BLE).

---

## 🔌 Hardware Configuration

### Pin Mapping (ESP32 DevKit)
*   **I2C Pins:** `SDA` (GPIO 21) & `SCL` (GPIO 22) -> connected to the TCA9548A multiplexer.
*   **GSR Pin:** Analog Input `GPIO 34` (connected directly to the Grove GSR sensor).
*   **Haptic / Alert:** Output `GPIO 25` (connected to buzzer / vibration motor).

### TCA9548A I2C Multiplexer Channels
*   **Channel 0 (Ch0):** MAX30102 PPG sensor (measures Heart Rate and SpO2).
*   **Channel 1 (Ch1):** MLX90614 Infrared thermometer (measures skin temperature).
*   **Channel 2 (Ch2):** SH1106 1.3" OLED display (128x64 pixels).

---

## 🧠 On-Device TinyML Inference

The firmware does not require heavy runtime libraries for inference. It executes a native C++ feedforward forward pass located in [ml_inference.cpp](./hydrocheck/ml_inference.cpp).

1.  **Normalization:** Scale raw signals using precomputed `StandardScaler` factors.
2.  **Rolling Buffer:** Keeps track of the last 10 samples (representing a 60-second window).
3.  **Feature Extraction:** Extracts 34 DSDT (Dynamic Statistics & Time-domain) features including slope, min, max, std dev, skewness, kurtosis, and correlation factors.
4.  **Neural Network Pass:** Computes matrix multiplications across layers (34 -> 64 -> 32 -> 16 -> 2) with ReLU activation, followed by stable Softmax to output classification probabilities.

---

## 📁 File Structure

*   [hydrocheck_ml.ino](./hydrocheck/hydrocheck_ml.ino): Core Arduino script with setup, sensor reading loops, Kalman filter, automated calibration, and BLE server.
*   [ml_inference.cpp](./hydrocheck/ml_inference.cpp) & [ml_inference.h](./hydrocheck/ml_inference.h): Zero-dependency native implementation of the neural network classifier.
*   [model_parameters.h](./hydrocheck/model_parameters.h): Exported weights and bias matrices from the Keras model.
*   [dehydration_model.h](./hydrocheck/dehydration_model.h): Alternative embedded model array.

---

## 🚀 Setup & Installation

### 1. Arduino IDE Setup
1. Download and open `hydrocheck_ml.ino` in the Arduino IDE.
2. Go to **Board Manager** and install support for **ESP32** (by Espressif Systems).
3. Select **ESP32 Dev Module** as the target board.

### 2. Library Dependencies
Install the following libraries using the Arduino Library Manager:
*   `SparkFun MAX3010x Pulse and Proximity Sensor Library`
*   `Adafruit MLX90614 Library`
*   `Adafruit SH110X`
*   `Adafruit GFX Library`

### 3. Flash Board
1. Connect the ESP32 to your PC using a micro-USB/USB-C cable.
2. Select the matching serial port.
3. Click the **Upload** arrow to compile and write the firmware.
