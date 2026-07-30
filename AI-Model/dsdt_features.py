import os
import numpy as np
from scipy.stats import skew, kurtosis

# SETTINGS

INPUT_DIR = "processed"
OUTPUT_DIR = "features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD WINDOW DATA

X = np.load(INPUT_DIR + "/X.npy")
y = np.load(INPUT_DIR + "/y.npy")

print("Loaded windows:", X.shape)

# Sensor index
GSR = 0
dGSR = 1
BPM = 2
HRV = 3
SkinT = 4
Strain = 5

# DSDT FEATURE FUNCTION

def extract_dsdt(window):

    features = []

    # GSR
    
    gsr = window[:, GSR]

    features.append(np.mean(gsr))
    features.append(np.std(gsr))
    features.append(np.min(gsr))
    features.append(np.max(gsr))

    features.append(gsr[-1]-gsr[0])

    features.append(np.polyfit(
        np.arange(len(gsr)),
        gsr,
        1
    )[0])

    features.append(np.trapezoid(gsr))

    features.append(skew(gsr))
    features.append(kurtosis(gsr))

    # dGSR

    dg = window[:, dGSR]

    features.append(np.mean(dg))
    features.append(np.std(dg))
    features.append(np.max(dg))

    # BPM

    bpm = window[:, BPM]

    features.append(np.mean(bpm))
    features.append(np.std(bpm))
    features.append(np.min(bpm))
    features.append(np.max(bpm))

    features.append(np.polyfit(
        np.arange(len(bpm)),
        bpm,
        1
    )[0])

    # HRV

    hrv = window[:, HRV]

    features.append(np.mean(hrv))
    features.append(np.std(hrv))
    features.append(np.min(hrv))
    features.append(np.max(hrv))

    features.append(np.polyfit(
        np.arange(len(hrv)),
        hrv,
        1
    )[0])

    # Skin Temperature

    temp = window[:, SkinT]

    features.append(np.mean(temp))
    features.append(np.std(temp))
    features.append(np.min(temp))
    features.append(np.max(temp))

    features.append(np.polyfit(
        np.arange(len(temp)),
        temp,
        1
    )[0])

    # Strain

    strain = window[:, Strain]

    features.append(np.mean(strain))
    features.append(np.std(strain))
    features.append(np.max(strain))

    # Cross Features

    features.append(np.corrcoef(gsr, bpm)[0,1])
    features.append(np.corrcoef(gsr, hrv)[0,1])
    features.append(np.corrcoef(gsr, temp)[0,1])
    features.append(np.corrcoef(hrv, bpm)[0,1])

    return np.nan_to_num(features)

# FEATURE EXTRACTION

feature_matrix = []

for sample in X:

    feature_matrix.append(
        extract_dsdt(sample)
    )

feature_matrix = np.array(feature_matrix)

print()

print("Feature Matrix:", feature_matrix.shape)


# SAVE


np.save(
    OUTPUT_DIR + "/X_features.npy",
    feature_matrix
)

np.save(
    OUTPUT_DIR + "/y_features.npy",
    y
)

print()

print("Finished")