import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# SETTINGS

FEATURE_DIR = "features"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD FEATURES

X = np.load(FEATURE_DIR + "/X_features.npy")
y = np.load(FEATURE_DIR + "/y_features.npy")

print("Feature Shape :", X.shape)
print("Label Shape   :", y.shape)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# MODEL

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(X.shape[1],)),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(np.unique(y)),
        activation="softmax"
    )

])

# COMPILE

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# CALLBACKS

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_DIR + "/best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=20,

    restore_best_weights=True

)

# TRAIN

history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=150,

    batch_size=16,

    callbacks=[checkpoint, early_stop],

    verbose=1

)

# SAVE FINAL MODEL

model.save(MODEL_DIR + "/final_model.keras")

# EVALUATE

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)

print("\nTest Accuracy :", accuracy)

# PREDICTIONS

pred = model.predict(X_test)

pred = np.argmax(pred, axis=1)

print()

print(classification_report(

    y_test,

    pred

))

# CONFUSION MATRIX

cm = confusion_matrix(

    y_test,

    pred

)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig(

    MODEL_DIR + "/confusion_matrix.png",

    dpi=300

)

plt.close()

# ACCURACY GRAPH

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.legend([

    "Training",

    "Validation"

])

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(

    MODEL_DIR + "/accuracy.png",

    dpi=300

)

plt.close()

# LOSS GRAPH

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.legend([

    "Training",

    "Validation"

])

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig(

    MODEL_DIR + "/loss.png",

    dpi=300

)

plt.close()

print("\nTraining Complete.")