# Occupancy Detection using Environmental Sensors and 1D CNN (FYP – EE3636)

## 📌 Overview

This project was developed as part of the **Final Year Engineering Project (EE3636)** in the BEng Computer Systems Engineering program at **Brunel University London**.

The system detects **human occupancy** using **time-series environmental sensor data** (temperature, humidity, gas levels, etc.) processed by a **1D Convolutional Neural Network (CNN)**. The model runs on a **Raspberry Pi**, enabling **on-device inference** without the need for cloud-based computation.

---

## 🎯 Objectives

- Develop a **low-cost, edge AI solution** for room occupancy detection
- Integrate multiple sensors (e.g., DHT11, MQ sensors, IR thermal imaging)
- Design and train a **1D CNN using TensorFlow/Keras**
- Deploy the trained model on a **Raspberry Pi with Python**
- Evaluate the model using **real sensor input + inference performance**

---

## ⚙️ Tech Stack

- 💻 **Raspberry Pi 5 Model B**
- 🧠 **TensorFlow / Keras** (1D CNN Model)
- 🧪 **Python** for data collection, preprocessing, and deployment
- 🌡️ Sensors: DHT11 (temperature/humidity), MQ-series (air quality), MLX90640 (thermal)
- 📦 **Edge Impulse** (optional: for TFLite conversion)
- 📈 Model evaluated using **accuracy, F1 score, and confusion matrix**

---

## 🧪 Features

- 🧠 Real-time 1D CNN inference on Raspberry Pi
- 🌡️ Multi-sensor integration (environmental + thermal)
- 💾 Sliding window data capture & buffering
- 📊 Visualization of classification results (e.g., via GUI or terminal)
- 📉 ROC-AUC & F1 Score metrics used for validation

---

## 🗂 Repo Structure

├── model/
│ ├── occupancy_1dcnn.h5 # Trained Keras model
│ ├── tflite_model.tflite # (optional) for edge deployment
├── src/
│ ├── main_pipeline.py # Sensor reading + inference
│ ├── preprocess.py # Sliding window + normalization
│ └── utils.py # Supporting functions
├── docs/
│ └── final_report.pdf # Final technical write-up
├── media/
│ └── system_architecture.png # System design diagram
└── README.md


---

## 📄 Report

📥 [Final Report](./docs/final_report.pdf)  
Contains:
- Problem definition, system architecture
- Sensor integration and data collection
- CNN architecture and training process
- Testing, evaluation, and deployment
- Discussion of energy efficiency and edge AI constraints

---

## 🖼️ Poster & Demo

📅 Presented on **7 May 2025** at Brunel EEE Poster Day  
🎥 *Demo video available upon request*

---

## 👤 Author

**Y. Benjamin Perez Condori**  
BEng Computer Systems Engineering  
Brunel University London  
📧 y.benjamin_pc@hotmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ybenjaminpc/)

---

## 🛡 Disclaimer

This repository contains **original work** created for educational and portfolio purposes only. No internal university documents or third-party IP are included.
