# 1. Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load and Preprocess Dataset
df = pd.read_csv(r"C:/Users/ybenj/Downloads/ALL_CSVS/merged_env_motion.csv")
df.drop_duplicates(inplace=True)
df.columns = df.columns.str.strip().str.lower()
df = df[df['motion'].notna()]
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
columns_to_drop = ['smoke', 'device']
if 'co' in df.columns:
    columns_to_drop.append('co')
df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') / 1e9
if 'light' in df.columns and df['light'].dtype == bool:
    df['light'] = df['light'].astype(int)
df.drop(columns=['hub', 'home'], inplace=True, errors='ignore')
target = 'motion'
df[target] = df[target].astype(int)
features = [col for col in df.columns if col != target]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32)

# 3. Create Sliding Windows
def create_windows(data, labels, window_size):
    X_windows, y_labels = [], []
    for i in range(len(data) - window_size):
        X_windows.append(data[i:i+window_size])
        y_labels.append(labels[i+window_size])
    return np.array(X_windows), np.array(y_labels)

TIME_STEPS = 100
BATCH_SIZE = 32
X_windowed, y_windowed = create_windows(X, y, TIME_STEPS)

# 4. TF Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_windowed, y_windowed))
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
total_batches = len(list(dataset))
train_size = int(0.8 * total_batches)
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

# 5. Compute Class Weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_windowed.astype(int)),
    y=y_windowed.astype(int)
)
class_weights = dict(enumerate(class_weights_array))

# 6. Define Conv1D Model (Deep Fixed)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(TIME_STEPS, len(features))),
    tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train with Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[early_stop], class_weight=class_weights)

# 8. Save Model (Keras + TFLite float32 + float16)
model.save("best_conv1d_model.keras")

# Float32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()
with open("model_float32.tflite", "wb") as f:
    f.write(tflite_model)

# Float16 TFLite
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model_fp16 = converter.convert()
with open("model_float16.tflite", "wb") as f:
    f.write(tflite_model_fp16)

# 9. Evaluate Model
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 10. Metrics & Visualization
y_true, y_pred_probs, y_pred_classes = [], [], []
for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch).flatten()
    y_true.extend(y_batch.numpy())
    y_pred_probs.extend(preds)
    y_pred_classes.extend((preds > 0.5).astype(int))

y_true = np.array(y_true).astype(int)
y_pred_probs = np.array(y_pred_probs)
y_pred_classes = np.array(y_pred_classes)

# Metrics
acc = accuracy_score(y_true, y_pred_classes)
prec = precision_score(y_true, y_pred_classes)
rec = recall_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes)
roc_auc = roc_auc_score(y_true, y_pred_probs)
logloss = log_loss(y_true, y_pred_probs)
cm = confusion_matrix(y_true, y_pred_classes)

# Print Results
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))
print(f"Accuracy:     {acc:.4f}")
print(f"Precision:    {prec:.4f}")
print(f"Recall:       {rec:.4f}")
print(f"F1 Score:     {f1:.4f}")
print(f"ROC-AUC:      {roc_auc:.4f}")
print(f"Log Loss:     {logloss:.4f}")

# Plots
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
