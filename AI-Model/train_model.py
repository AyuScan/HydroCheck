import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

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

# 1. INTEGRATED TRAINING METRICS DASHBOARD
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Subplot
ax1.plot(history.history["accuracy"], label="Training", color="#1f77b4", lw=2)
ax1.plot(history.history["val_accuracy"], label="Validation", color="#ff7f0e", lw=2)
ax1.set_title("Model Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="lower right")

# Loss Subplot
ax2.plot(history.history["loss"], label="Training", color="#1f77b4", lw=2)
ax2.plot(history.history["val_loss"], label="Validation", color="#ff7f0e", lw=2)
ax2.set_title("Model Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(MODEL_DIR + "/training_metrics_dashboard.png", dpi=300)
plt.close()


# 2. MULTI-CLASS ROC CURVE
# Binarize the labels for the test set
unique_classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=unique_classes)
n_classes = len(unique_classes)

# Predict class probabilities
pred_probs = model.predict(X_test)

plt.figure(figsize=(8, 6))
colors = ["#4caf50", "#ff9800", "#f44336"]  # Low risk (green), Moderate risk (orange), High risk (red)
label_names = ["Low Risk", "Moderate Risk", "High Risk"] # Human readable labels matching unique label indices

for i in range(n_classes):
    # If binary classification fallback
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="#2196f3", lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        break
    else:
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        label_text = label_names[i] if i < len(label_names) else f"Class {unique_classes[i]}"
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{label_text} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) - Multi-Class")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig(MODEL_DIR + "/roc_curve.png", dpi=300)
plt.close()


# 3. FEATURE CORRELATION HEATMAP
try:
    df_raw = pd.read_csv("hydrocheck_dataset.csv")
    
    # Rename trailing-comma target column to Label if Unnamed
    LABEL_COL = "Label"
    if LABEL_COL not in df_raw.columns:
        unnamed_cols = [c for c in df_raw.columns if "Unnamed" in c]
        if unnamed_cols:
            df_raw = df_raw.rename(columns={unnamed_cols[0]: LABEL_COL})
        else:
            df_raw = df_raw.rename(columns={df_raw.columns[-1]: LABEL_COL})

    # Drop irrelevant columns
    drop_cols = ["Timestamp (s)", "Calib", "Risk"]
    df_raw = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])
    
    # Label encode the non-numeric target column if it's there
    if LABEL_COL in df_raw.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_raw[LABEL_COL] = le.fit_transform(df_raw[LABEL_COL].astype(str))

    corr_matrix = df_raw.corr()
    
    plt.figure(figsize=(8, 6))
    # We can plot a clean correlation matrix using matplotlib
    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation Coefficient")
    
    # Set tick labels
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Annotate correlation numbers
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", 
                     color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                     
    plt.title("Sensor Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(MODEL_DIR + "/feature_correlation.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Skipped correlation heatmap generation: {e}")


# 4. PERMUTATION FEATURE IMPORTANCE (34 DSDT Features)
FEATURE_NAMES = [
    "GSR_mean", "GSR_std", "GSR_min", "GSR_max", "GSR_diff", "GSR_slope", "GSR_area", "GSR_skew", "GSR_kurt",
    "dGSR_mean", "dGSR_std", "dGSR_max",
    "BPM_mean", "BPM_std", "BPM_min", "BPM_max", "BPM_slope",
    "HRV_mean", "HRV_std", "HRV_min", "HRV_max", "HRV_slope",
    "SkinT_mean", "SkinT_std", "SkinT_min", "SkinT_max", "SkinT_slope",
    "Strain_mean", "Strain_std", "Strain_max",
    "Corr_GSR_BPM", "Corr_GSR_HRV", "Corr_GSR_SkinT", "Corr_HRV_BPM"
]

print("\nCalculating Permutation Feature Importance (this might take a moment)...")
try:
    baseline_preds = model.predict(X_test, verbose=0)
    baseline_preds_class = np.argmax(baseline_preds, axis=1)
    baseline_accuracy = np.mean(baseline_preds_class == y_test)

    importances = []
    for col_idx in range(X_test.shape[1]):
        # Shuffle values of this column in the test set
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled[:, col_idx])
        
        shuffled_preds = model.predict(X_test_shuffled, verbose=0)
        shuffled_preds_class = np.argmax(shuffled_preds, axis=1)
        shuffled_accuracy = np.mean(shuffled_preds_class == y_test)
        
        importance = baseline_accuracy - shuffled_accuracy
        importances.append(importance)

    # Plot top 15 most important features
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 8))
    
    # We display up to top 15 features for clarity on a horizontal bar chart
    top_n = min(15, len(importances))
    top_indices = sorted_idx[-top_n:]
    
    names_to_plot = [FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Feature_{idx}" for idx in top_indices]
    importances_to_plot = [importances[idx] for idx in top_indices]
    
    plt.barh(names_to_plot, importances_to_plot, color="#3f51b5")
    plt.xlabel("Accuracy Decrease when Permuted")
    plt.title("Top 15 Most Influential Features (Permutation Importance)")
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig(MODEL_DIR + "/feature_importance.png", dpi=300)
    plt.close()
    print("Feature importance graph saved successfully.")
except Exception as e:
    print(f"Skipped feature importance generation: {e}")

# 5. t-SNE PROJECTION PLOT (Cluster Analysis)
print("\nCalculating t-SNE Projection (this might take a moment)...")
try:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    colors = ['#4caf50', '#ff9800'] # Colors for Low (green) and Moderate (orange) risk
    labels = ['Low Risk', 'Moderate Risk']

    for class_idx in np.unique(y):
        mask = (y == class_idx)
        plt.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1],
            c=colors[class_idx % len(colors)], label=labels[class_idx % len(labels)],
            alpha=0.8, edgecolors='none', s=45
        )

    plt.title("t-SNE Visualization of Dehydration Feature Space", fontsize=12, fontweight='bold')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(MODEL_DIR + "/research_tsne_clusters.png", dpi=300)
    plt.close()
    print("t-SNE plot saved in model/research_tsne_clusters.png")
except Exception as e:
    print(f"Skipped t-SNE plot generation: {e}")


# 6. FEATURE DISTRIBUTION SHIFT (KDE / Density Plot)
try:
    df_raw = pd.read_csv("hydrocheck_dataset.csv")
    
    # Rename trailing-comma target column to Label if Unnamed
    LABEL_COL = "Label"
    if LABEL_COL not in df_raw.columns:
        unnamed_cols = [c for c in df_raw.columns if "Unnamed" in c]
        if unnamed_cols:
            df_raw = df_raw.rename(columns={unnamed_cols[0]: LABEL_COL})
        else:
            df_raw = df_raw.rename(columns={df_raw.columns[-1]: LABEL_COL})

    plt.figure(figsize=(8, 5))
    try:
        import seaborn as sns
        sns.kdeplot(data=df_raw, x="GSR", hue=LABEL_COL, fill=True, common_norm=False, palette="Set1", alpha=0.5, linewidth=2)
    except ImportError:
        # Fallback to matplotlib histogram overlay if seaborn isn't installed
        for label_val in df_raw[LABEL_COL].unique():
            subset = df_raw[df_raw[LABEL_COL] == label_val]
            plt.hist(subset["GSR"], bins=30, alpha=0.5, density=True, label=str(label_val))
            
    plt.title("Distribution of Galvanic Skin Response (GSR) by Hydration Status", fontsize=12, fontweight='bold')
    plt.xlabel("GSR Conductance (Normalized/Raw)")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(MODEL_DIR + "/research_gsr_density.png", dpi=300)
    plt.close()
    print("Density plot saved in model/research_gsr_density.png")
except Exception as e:
    print(f"Skipped density plot generation: {e}")


# 7. PHYSIOLOGICAL SENSOR TIME-SERIES TRANSITION
try:
    df_raw = pd.read_csv("hydrocheck_dataset.csv")
    
    # Rename trailing-comma target column to Label if Unnamed
    LABEL_COL = "Label"
    if LABEL_COL not in df_raw.columns:
        unnamed_cols = [c for c in df_raw.columns if "Unnamed" in c]
        if unnamed_cols:
            df_raw = df_raw.rename(columns={unnamed_cols[0]: LABEL_COL})
        else:
            df_raw = df_raw.rename(columns={df_raw.columns[-1]: LABEL_COL})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot GSR
    ax1.plot(df_raw["Timestamp (s)"] / 60.0, df_raw["GSR"], color="#2ca02c", lw=2)
    ax1.set_ylabel("GSR (Conductance)")
    ax1.set_title("Physiological Telemetry Over Time", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Plot BPM
    ax2.plot(df_raw["Timestamp (s)"] / 60.0, df_raw["BPM"], color="#d62728", lw=2)
    ax2.set_ylabel("Heart Rate (BPM)")
    ax2.set_xlabel("Time (Minutes)")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Highlight Transition Phase (Example: after 10 minutes)
    ax1.axvspan(10, df_raw["Timestamp (s)"].max()/60.0, color='red', alpha=0.1, label='Moderate Dehydration Zone')
    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(MODEL_DIR + "/research_sensor_timeseries.png", dpi=300)
    plt.close()
    print("Time-series transition plot saved in model/research_sensor_timeseries.png")
except Exception as e:
    print(f"Skipped time-series plot generation: {e}")

print("\nTraining Complete.")