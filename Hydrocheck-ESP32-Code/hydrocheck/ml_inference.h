#ifndef ML_INFERENCE_H
#define ML_INFERENCE_H

#include <Arduino.h>
#include <math.h>

// This file contains the exact float weights and biases exported directly from best_model.keras
#include "model_parameters.h"

#define NUMBER_OF_INPUTS 34
#define NUMBER_OF_OUTPUTS 2

namespace DehydrationML {

    // StandardScaler parameters (extracted from scaler.pkl)
    // Order of features: [GSR, dGSR, BPM, HRV, SkinT, Strain]
    const float SCALER_MEAN[6]  = { 1829.15276f, 10.85384f, 76.95143f, 140.97483f, 34.81943f, 6.79360f };
    const float SCALER_SCALE[6] = {  386.57818f,  24.22391f,  8.60475f,  96.37589f,  1.31215f, 11.29974f };

    // Rolling window settings
    const int WINDOW_SIZE = 10;
    const int FEATURE_COUNT = 6;

    // Buffer to hold 10 historical samples of 6 scaled features
    extern float windowBuffer[WINDOW_SIZE][FEATURE_COUNT];
    extern int bufferCount;

    // Add a new raw sample, normalize it using the StandardScaler, and shift into window
    void addSample(float gsr, float dgsr, float bpm, float hrv, float skint, float strain);

    // Compute the 34 DSDT features from the current normalized window
    void extractFeatures(float* featuresOut);

    // Run inference using a pure C++ implementation of the neural network
    // Returns the class index (0 = Low risk, 1 = Moderate risk)
    // and populates probabilities: probLowRisk and probModRisk
    int predict(float& probLowRisk, float& probModRisk);
}

#endif // ML_INFERENCE_H
