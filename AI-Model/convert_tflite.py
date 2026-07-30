import os
import tensorflow as tf

# SETTINGS

MODEL_PATH = "model/best_model.keras"

OUTPUT_DIR = "tflite"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD MODEL

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

# FLOAT32 TFLITE

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(
    OUTPUT_DIR + "/dehydration_model.tflite",
    "wb"
) as f:
    f.write(tflite_model)

print("Float model saved.")

# INT8 QUANTIZATION

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# Representative dataset
import numpy as np

X = np.load("features/X_features.npy")

def representative_dataset():

    for i in range(min(100, len(X))):

        yield [
            X[i:i+1].astype(np.float32)
        ]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant = converter.convert()

with open(
    OUTPUT_DIR + "/dehydration_model_int8.tflite",
    "wb"
) as f:
    f.write(tflite_quant)

print("INT8 model saved.")