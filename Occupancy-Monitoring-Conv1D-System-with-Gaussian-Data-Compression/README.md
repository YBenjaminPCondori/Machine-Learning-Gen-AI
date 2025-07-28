
# 🧠 Final Year Project – Edge AI Occupancy Detection using Conv1D

This repository contains the code, model, and evaluation results for my **final-year BEng Computer Systems Engineering project**, focused on **real-time occupancy detection** using environmental sensor data and a **1D Convolutional Neural Network (Conv1D)**. The final trained model is **quantized for edge deployment** on devices like the Raspberry Pi.

---

## 📌 Project Summary

The system detects **occupancy (motion)** using a series of inexpensive sensors (e.g. DHT11, PM2.5, infrared). It transforms raw sensor readings into **time-series windows** and uses a deep Conv1D neural network to classify whether the room is **occupied or empty**.

✅ Designed for **resource-constrained edge devices**  
✅ Model trained and optimized in **TensorFlow**, converted to **TensorFlow Lite (TFLite)**  
✅ Evaluated with metrics suitable for deployment and academic research

---

## 🎯 Objectives

- Predict motion/occupancy from multi-sensor time-series data  
- Optimize a deep learning model for **TinyML / Edge AI deployment**  
- Evaluate model performance using real-world metrics (F1, ROC-AUC, confusion matrix)  
- Ensure compatibility with low-power hardware (e.g., Raspberry Pi, microcontrollers)

---

## 🗂️ Dataset

CSV file:  
```
merged_env_motion.csv
```

Must include:
- Environmental features (e.g. `temperature`, `humidity`, `alcohol`, `pm2.5`)
- Binary target: `motion` (0 = no motion, 1 = motion)
- Timestamps and optional features (`light`, `smoke`, etc.) are handled automatically.

---

## 📈 Model Architecture

```
Input (100 timesteps, N features)
↓
Conv1D (64 filters) + BatchNorm + MaxPool
↓
Conv1D (128 filters) + BatchNorm + MaxPool
↓
Conv1D (256 filters) + BatchNorm + MaxPool
↓
Flatten → Dense(128) → Dropout
↓
Dense(64) → Dropout
↓
Output: Dense(1, sigmoid)
```

[Neural Network Architecture of CNN](https://i.imgur.com/or5WaAS.png)

Trained with:
- Binary crossentropy loss
- Adam optimizer
- Class balancing
- Early stopping on validation loss

---

## 🧪 Evaluation Metrics

All metrics are printed after training:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ ROC-AUC
- ✅ Log Loss
- ✅ Confusion Matrix (Heatmap)
- ✅ ROC Curve

---

## 💾 Model Outputs

- `best_conv1d_model.keras` – Native Keras format
- `model_float32.tflite` – Full precision for testing
- `model_float16.tflite` – Optimized for deployment (Edge AI / Raspberry Pi)

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

---

## ▶️ How to Run

```bash
python occupancy_conv1d.py
```

Ensure the CSV file is in the correct path and named `merged_env_motion.csv`.

---

## 🔗 Related Work

- 📄 [Dissertation PDF](#) *(Upload and update the link when ready)*
- 📊 [Project Poster](#) *(Optional)*
- 💻 Runs on: Raspberry Pi 5 Model B (on-device inference tested)

---

## 🛠️ Deployment Notes

- Model supports conversion to TFLite (float16) for **TinyML applications**
- Ideal for integration with sensor platforms using **Python + TensorFlow Lite Runtime**
- May be extended to real-time inference with camera + sensor fusion

---

## 📚 Citation

If you use this in academic work:

```
Y. B. Perez Condori, "Edge AI Occupancy Detection Using Conv1D and Environmental Sensor Data," Final Year Project, Brunel University London, 2025.
```

---

## 📄 License

MIT License – use, share, or extend freely.
