
<h1 align="center">📡 Final Year Project – Edge AI Occupancy Detection using Conv1D</h1>

<p align="center">
  <em>This repository presents the complete pipeline for deploying an Edge AI occupancy detection model using environmental sensor data and a lightweight Conv1D neural network.</em>
</p>

<p align="center">
  <strong>BEng Computer Systems Engineering | Brunel University London | 2025</strong>
</p>

<hr/>

## 📌 Project Summary

This system detects **occupancy (motion)** using inexpensive sensors (DHT11, PM2.5, IR) and processes the data into **time-series windows** for classification via a deep **1D Convolutional Neural Network (Conv1D)**.

- ✅ Built for **resource-constrained edge devices**
- ✅ Trained using **TensorFlow**, optimized with **TFLite** for deployment
- ✅ Evaluated with real-world ML metrics (F1, ROC-AUC, confusion matrix)

---

## 🎯 Objectives

<p align="center">
  <img src="../system%20design%20occupancy%20monitoring%20TINYML.png" alt="System Overview" width="600"/>
</p>

- Predict occupancy from multi-sensor time-series data  
- Optimize for **TinyML / Edge AI** deployment  
- Evaluate using F1 Score, ROC-AUC, and Confusion Matrix  
- Ensure compatibility with low-power platforms like **Raspberry Pi**

---

## 🗂️ Dataset Format

File required: `merged_env_motion.csv`

```csv
timestamp, temperature, humidity, alcohol, pm2.5, motion
2024-01-01 00:00, 21.5, 45.2, 0.03, 12, 1
...
```

- 🟢 Features: `temperature`, `humidity`, `alcohol`, `pm2.5`
- 🔵 Label: `motion` (binary: 0 = no motion, 1 = motion)
- ⏱️ Timestamp used for window slicing (optional)

---

## 🧠 Model Architecture

<p align="center">
  <img src="../NN%20Architecture.png" alt="Conv1D Model Architecture" width="500"/>
</p>

- Loss Function: **Binary Crossentropy**
- Optimizer: **Adam**
- Training techniques: **Class balancing**, **Early stopping**

---

## 📊 Evaluation Metrics

Model performance is assessed using:

- ✔️ Accuracy, Precision, Recall  
- ✔️ **F1 Score**, **ROC-AUC**, Log Loss  
- ✔️ **Confusion Matrix** (heatmap)  
- ✔️ **ROC Curve**

---

## 💾 Model Outputs

| File | Format | Purpose |
|------|--------|---------|
| `best_conv1d_model.keras` | Keras | Native model checkpoint |
| `model_float32.tflite` | TFLite | High-precision testing |
| `model_float8.tflite` | TFLite | Deployment-ready, 8-bit quantized |

---

## 📦 Requirements

Install the following packages:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

📌 *Pandas and NumPy are essential for preprocessing and data handling.*

---

## ▶️ Running the Project

1. Ensure you're using a Linux-based SBC (e.g. Raspberry Pi) or a microcontroller with support for ML inference.
2. Place your dataset file as `merged_env_motion.csv` in the working directory.
3. Run:

```bash
python occupancy_conv1d.py
```

---

## 🔗 Related Work

- 📄 [Project Poster](../poster.pdf)  
- 🧪 Tested on: **Raspberry Pi 5 Model B** with on-device inference

---

## 🚀 Deployment Notes

- Compatible with **TensorFlow Lite Micro** and **TFLite Runtime**
- Suitable for **sensor fusion**, **real-time inference**, and **low-power ML**
- Expandable to include **thermal imaging**, **infrared**, or **camera input**

---

## 📚 Citation

```
Y. B. Perez Condori, "Edge AI Occupancy Detection Using Conv1D and Environmental Sensor Data," Final Year Project, Brunel University of London, 2025.
```

---

## 📄 License

MIT License – Free to use, modify, and distribute.
