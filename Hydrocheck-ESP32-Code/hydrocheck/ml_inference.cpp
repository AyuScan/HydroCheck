#include "ml_inference.h"

namespace DehydrationML {

    float windowBuffer[WINDOW_SIZE][FEATURE_COUNT] = {0.0f};
    int bufferCount = 0;

    void addSample(float gsr, float dgsr, float bpm, float hrv, float skint, float strain) {
        // 1. Normalize raw inputs using StandardScaler parameters
        float normGsr    = (gsr - SCALER_MEAN[0]) / SCALER_SCALE[0];
        float normDgsr   = (dgsr - SCALER_MEAN[1]) / SCALER_SCALE[1];
        float normBpm    = (bpm - SCALER_MEAN[2]) / SCALER_SCALE[2];
        float normHrv    = (hrv - SCALER_MEAN[3]) / SCALER_SCALE[3];
        float normSkint  = (skint - SCALER_MEAN[4]) / SCALER_SCALE[4];
        float normStrain = (strain - SCALER_MEAN[5]) / SCALER_SCALE[5];

        // 2. Shift history if buffer is full
        if (bufferCount >= WINDOW_SIZE) {
            for (int i = 0; i < WINDOW_SIZE - 1; i++) {
                for (int f = 0; f < FEATURE_COUNT; f++) {
                    windowBuffer[i][f] = windowBuffer[i + 1][f];
                }
            }
            // Put in the last slot
            windowBuffer[WINDOW_SIZE - 1][0] = normGsr;
            windowBuffer[WINDOW_SIZE - 1][1] = normDgsr;
            windowBuffer[WINDOW_SIZE - 1][2] = normBpm;
            windowBuffer[WINDOW_SIZE - 1][3] = normHrv;
            windowBuffer[WINDOW_SIZE - 1][4] = normSkint;
            windowBuffer[WINDOW_SIZE - 1][5] = normStrain;
        } else {
            // Fill up sequentially
            windowBuffer[bufferCount][0] = normGsr;
            windowBuffer[bufferCount][1] = normDgsr;
            windowBuffer[bufferCount][2] = normBpm;
            windowBuffer[bufferCount][3] = normHrv;
            windowBuffer[bufferCount][4] = normSkint;
            windowBuffer[bufferCount][5] = normStrain;
            bufferCount++;
        }
    }

    // Helper functions for stats
    float getMean(const float* x, int n) {
        float sum = 0;
        for (int i = 0; i < n; i++) sum += x[i];
        return sum / n;
    }

    float getStd(const float* x, int n, float mean) {
        float sumSqDiff = 0;
        for (int i = 0; i < n; i++) {
            float diff = x[i] - mean;
            sumSqDiff += diff * diff;
        }
        return sqrtf(sumSqDiff / n); // Population standard deviation (ddof=0)
    }

    float getMin(const float* x, int n) {
        float minVal = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] < minVal) minVal = x[i];
        }
        return minVal;
    }

    float getMax(const float* x, int n) {
        float maxVal = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] > maxVal) maxVal = x[i];
        }
        return maxVal;
    }

    float getSlope(const float* x, int n) {
        // Linear regression slope for indices 0..N-1
        // N = 10 -> sum(i) = 45, sum(i^2) = 285, Denom = N*sum(i^2) - sum(i)^2 = 825
        float sumX = 0;
        float sumIX = 0;
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumIX += i * x[i];
        }
        return (10.0f * sumIX - 45.0f * sumX) / 825.0f;
    }

    float getTrapezoid(const float* x, int n) {
        // np.trapezoid(x) returns the integration over grid spacing dx=1
        // trapz = 0.5 * (x[0] + x[n-1]) + sum_{i=1..n-2}(x[i])
        float sum = 0.5f * (x[0] + x[n - 1]);
        for (int i = 1; i < n - 1; i++) {
            sum += x[i];
        }
        return sum;
    }

    float getSkewness(const float* x, int n, float mean, float std) {
        if (std < 1e-6f) return 0.0f;
        float sum = 0;
        for (int i = 0; i < n; i++) {
            float z = (x[i] - mean) / std;
            sum += z * z * z;
        }
        return sum / n;
    }

    float getKurtosis(const float* x, int n, float mean, float std) {
        if (std < 1e-6f) return 0.0f;
        float sum = 0;
        for (int i = 0; i < n; i++) {
            float z = (x[i] - mean) / std;
            sum += z * z * z * z;
        }
        return (sum / n) - 3.0f; // Excess kurtosis (Fisher's definition)
    }

    float getCorrelation(const float* x, const float* y, int n, float meanX, float stdX, float meanY, float stdY) {
        if (stdX < 1e-6f || stdY < 1e-6f) return 0.0f;
        float covariance = 0;
        for (int i = 0; i < n; i++) {
            covariance += (x[i] - meanX) * (y[i] - meanY);
        }
        covariance /= n;
        return covariance / (stdX * stdY);
    }

    void extractFeatures(float* fOut) {
        // Extract 1D arrays for each feature from the window
        float gsr[WINDOW_SIZE];
        float dgsr[WINDOW_SIZE];
        float bpm[WINDOW_SIZE];
        float hrv[WINDOW_SIZE];
        float skint[WINDOW_SIZE];
        float strain[WINDOW_SIZE];

        for (int i = 0; i < WINDOW_SIZE; i++) {
            gsr[i]    = windowBuffer[i][0];
            dgsr[i]   = windowBuffer[i][1];
            bpm[i]    = windowBuffer[i][2];
            hrv[i]    = windowBuffer[i][3];
            skint[i]  = windowBuffer[i][4];
            strain[i] = windowBuffer[i][5];
        }

        // Calculate statistics
        float meanGsr   = getMean(gsr, WINDOW_SIZE);
        float stdGsr    = getStd(gsr, WINDOW_SIZE, meanGsr);
        float minGsr    = getMin(gsr, WINDOW_SIZE);
        float maxGsr    = getMax(gsr, WINDOW_SIZE);

        float meanDgsr  = getMean(dgsr, WINDOW_SIZE);
        float stdDgsr   = getStd(dgsr, WINDOW_SIZE, meanDgsr);
        float maxDgsr   = getMax(dgsr, WINDOW_SIZE);

        float meanBpm   = getMean(bpm, WINDOW_SIZE);
        float stdBpm    = getStd(bpm, WINDOW_SIZE, meanBpm);
        float minBpm    = getMin(bpm, WINDOW_SIZE);
        float maxBpm    = getMax(bpm, WINDOW_SIZE);

        float meanHrv   = getMean(hrv, WINDOW_SIZE);
        float stdHrv    = getStd(hrv, WINDOW_SIZE, meanHrv);
        float minHrv    = getMin(hrv, WINDOW_SIZE);
        float maxHrv    = getMax(hrv, WINDOW_SIZE);

        float meanSkint = getMean(skint, WINDOW_SIZE);
        float stdSkint  = getStd(skint, WINDOW_SIZE, meanSkint);
        float minSkint  = getMin(skint, WINDOW_SIZE);
        float maxSkint  = getMax(skint, WINDOW_SIZE);

        float meanStrain= getMean(strain, WINDOW_SIZE);
        float stdStrain = getStd(strain, WINDOW_SIZE, meanStrain);
        float maxStrain = getMax(strain, WINDOW_SIZE);

        int idx = 0;

        // --- 1. GSR features (9 features) ---
        fOut[idx++] = meanGsr;
        fOut[idx++] = stdGsr;
        fOut[idx++] = minGsr;
        fOut[idx++] = maxGsr;
        fOut[idx++] = gsr[WINDOW_SIZE - 1] - gsr[0]; // diff
        fOut[idx++] = getSlope(gsr, WINDOW_SIZE);
        fOut[idx++] = getTrapezoid(gsr, WINDOW_SIZE);
        fOut[idx++] = getSkewness(gsr, WINDOW_SIZE, meanGsr, stdGsr);
        fOut[idx++] = getKurtosis(gsr, WINDOW_SIZE, meanGsr, stdGsr);

        // --- 2. dGSR features (3 features) ---
        fOut[idx++] = meanDgsr;
        fOut[idx++] = stdDgsr;
        fOut[idx++] = maxDgsr;

        // --- 3. BPM features (5 features) ---
        fOut[idx++] = meanBpm;
        fOut[idx++] = stdBpm;
        fOut[idx++] = minBpm;
        fOut[idx++] = maxBpm;
        fOut[idx++] = getSlope(bpm, WINDOW_SIZE);

        // --- 4. HRV features (5 features) ---
        fOut[idx++] = meanHrv;
        fOut[idx++] = stdHrv;
        fOut[idx++] = minHrv;
        fOut[idx++] = maxHrv;
        fOut[idx++] = getSlope(hrv, WINDOW_SIZE);

        // --- 5. SkinT features (5 features) ---
        fOut[idx++] = meanSkint;
        fOut[idx++] = stdSkint;
        fOut[idx++] = minSkint;
        fOut[idx++] = maxSkint;
        fOut[idx++] = getSlope(skint, WINDOW_SIZE);

        // --- 6. Strain features (3 features) ---
        fOut[idx++] = meanStrain;
        fOut[idx++] = stdStrain;
        fOut[idx++] = maxStrain;

        // --- 7. Cross correlation features (4 features) ---
        fOut[idx++] = getCorrelation(gsr, bpm, WINDOW_SIZE, meanGsr, stdGsr, meanBpm, stdBpm);
        fOut[idx++] = getCorrelation(gsr, hrv, WINDOW_SIZE, meanGsr, stdGsr, meanHrv, stdHrv);
        fOut[idx++] = getCorrelation(gsr, skint, WINDOW_SIZE, meanGsr, stdGsr, meanSkint, stdSkint);
        fOut[idx++] = getCorrelation(hrv, bpm, WINDOW_SIZE, meanHrv, stdHrv, meanBpm, stdBpm);
    }

    int predict(float& probLowRisk, float& probModRisk) {
        if (bufferCount < WINDOW_SIZE) {
            probLowRisk = 1.0f;
            probModRisk = 0.0f;
            return 0;
        }

        // 1. Extract 34 float features
        float features[NUMBER_OF_INPUTS];
        extractFeatures(features);

        // 2. Forward pass: Layer 1 (Dense 34 -> 64, ReLU)
        float layer1[64];
        for (int c = 0; c < 64; c++) {
            float val = biasL1[c];
            for (int r = 0; r < 34; r++) {
                val += features[r] * weightsL1[r * 64 + c];
            }
            layer1[c] = val > 0.0f ? val : 0.0f; // ReLU
        }

        // Forward pass: Layer 2 (Dense 64 -> 32, ReLU)
        float layer2[32];
        for (int c = 0; c < 32; c++) {
            float val = biasL2[c];
            for (int r = 0; r < 64; r++) {
                val += layer1[r] * weightsL2[r * 32 + c];
            }
            layer2[c] = val > 0.0f ? val : 0.0f; // ReLU
        }

        // Forward pass: Layer 3 (Dense 32 -> 16, ReLU)
        float layer3[16];
        for (int c = 0; c < 16; c++) {
            float val = biasL3[c];
            for (int r = 0; r < 32; r++) {
                val += layer2[r] * weightsL3[r * 16 + c];
            }
            layer3[c] = val > 0.0f ? val : 0.0f; // ReLU
        }

        // Forward pass: Layer 4 (Dense 16 -> 2, Softmax)
        float logits[2];
        float maxLogit = -1e9f;
        for (int c = 0; c < 2; c++) {
            float val = biasL4[c];
            for (int r = 0; r < 16; r++) {
                val += layer3[r] * weightsL4[r * 2 + c];
            }
            logits[c] = val;
            if (val > maxLogit) maxLogit = val;
        }

        // Softmax with numerical stability
        float sumExp = 0.0f;
        for (int c = 0; c < 2; c++) {
            logits[c] = expf(logits[c] - maxLogit);
            sumExp += logits[c];
        }

        probLowRisk = logits[0] / sumExp;
        probModRisk = logits[1] / sumExp;

        return (probModRisk > probLowRisk) ? 1 : 0;
    }
}
