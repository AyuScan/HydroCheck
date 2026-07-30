import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder

# SETTINGS

DATASET = "hydrocheck_dataset.csv"

WINDOW_SIZE = 10          # 10 samples = 60 seconds
STEP_SIZE = 1             # Sliding window

FEATURES = [
    "GSR",
    "dGSR",
    "BPM",
    "HRV",
    "SkinT",
    "Strain"
]

LABEL = "Label"

OUTPUT_DIR = "processed"

# CREATE OUTPUT DIRECTORY

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DATASET

print("Loading dataset...")

df = pd.read_csv(DATASET)

print(df.head())

# REMOVE UNUSED COLUMNS


drop_cols = []

for c in ["Risk", "Calib", "Timestamp (s)"]:
    if c in df.columns:
        drop_cols.append(c)

df = df.drop(columns=drop_cols)

# REMOVE MISSING VALUES

df = df.dropna()

# LABEL ENCODING

encoder = LabelEncoder()

df[LABEL] = encoder.fit_transform(df[LABEL])

print("\nClasses:")

for i, c in enumerate(encoder.classes_):
    print(i, c)

joblib.dump(encoder, OUTPUT_DIR + "/label_encoder.pkl")

# NORMALIZE FEATURES

scaler = StandardScaler()

df[FEATURES] = scaler.fit_transform(df[FEATURES])

joblib.dump(scaler, OUTPUT_DIR + "/scaler.pkl")

# CREATE SLIDING WINDOWS

X = []
y = []

for start in range(0, len(df)-WINDOW_SIZE+1, STEP_SIZE):

    end = start + WINDOW_SIZE

    window = df.iloc[start:end]

    X.append(window[FEATURES].values)

    y.append(window[LABEL].iloc[-1])

X = np.array(X)

y = np.array(y)

print()

print("Input Shape :", X.shape)

print("Labels Shape:", y.shape)

# SAVE

np.save(OUTPUT_DIR + "/X.npy", X)

np.save(OUTPUT_DIR + "/y.npy", y)

print()

print("Done.")