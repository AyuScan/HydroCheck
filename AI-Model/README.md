# Hydrocheck AI Model Training Pipeline

This directory contains the code to convert raw time-series data, extract high-dimensional features, train classification models, and export weights for ESP32 on-device edge inference.

---

## 📊 Pipeline Flow

```
[Raw logs: datas.txt] 
   │
   ▼ (csv-transformer.py)
[Structured CSV: hydrocheck_dataset.csv]
   │
   ▼ (preprocessing.py)
[Cleaned, Scaled & Windowed Arrays: X.npy & y.npy]
   │
   ▼ (dsdt_features.py)
[34 DSDT Features Matrix: X_features.npy & y_features.npy]
   │
   ▼ (train_model.py)
[Trained Keras Model: best_model.keras]
   │
   ▼ (convert_tflite.py)
[Quantized TFLite: dehydration_model_int8.tflite] -> Export to model_parameters.h
```

---

## 📁 File Structure

*   [csv-transformer.py](./csv-transformer.py): Transforms space-separated telemetry values to standard CSV columns.
*   [preprocessing.py](./preprocessing.py): Normalizes the dataset using `StandardScaler` and constructs 10-sample sliding windows (representing 60s windows).
*   [dsdt_features.py](./dsdt_features.py): Computes 34 statistical time-domain and correlation features per window.
*   [train_model.py](./train_model.py): Defines the neural network topology, fits validation arrays, outputs learning curve figures, and exports `best_model.keras`.
*   [convert_tflite.py](./convert_tflite.py): Converts Keras models to float32 and INT8 quantized TFLite representations.

---

## 📈 Feature Engineering Details
The **34 DSDT features** are extracted from each 10-sample window over 6 input metrics (`GSR`, `dGSR`, `BPM`, `HRV`, `SkinT`, `Strain`):
*   **GSR (9):** Mean, Standard Deviation, Min, Max, Delta, Regression Slope, Trapezoidal Area, Skewness, Kurtosis.
*   **dGSR (3):** Mean, Standard Deviation, Max.
*   **BPM (5):** Mean, Standard Deviation, Min, Max, Regression Slope.
*   **HRV (5):** Mean, Standard Deviation, Min, Max, Regression Slope.
*   **SkinT (5):** Mean, Standard Deviation, Min, Max, Regression Slope.
*   **Strain (3):** Mean, Standard Deviation, Max.
*   **Cross Correlation (4):** Pearson Coefficients for GSR-BPM, GSR-HRV, GSR-SkinT, HRV-BPM.

---

## 🚀 Execution Guide

1. **Install Requirements:**
   ```bash
   pip install numpy scipy pandas scikit-learn tensorflow matplotlib joblib
   ```
2. **Convert raw telemetry data:**
   ```bash
   python csv-transformer.py
   ```
3. **Preprocess and partition windows:**
   ```bash
   python preprocessing.py
   ```
4. **Compute features:**
   ```bash
   python dsdt_features.py
   ```
5. **Train the classification model:**
   ```bash
   python train_model.py
   ```
6. **Quantize and convert model:**
   ```bash
   python convert_tflite.py
   ```
