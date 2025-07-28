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
