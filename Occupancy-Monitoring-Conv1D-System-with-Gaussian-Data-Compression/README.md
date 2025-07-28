
#  Final Year Project – Edge AI Occupancy Detection using Conv1D

This repository contains the code, model, and evaluation results for my **final-year BEng Computer Systems Engineering project**, focused on **real-time occupancy detection** using environmental sensor data and a **1D Convolutional Neural Network (Conv1D)**. The final trained model is **quantized for edge deployment** on devices like the Raspberry Pi.

---

## 📌 Project Summary

The system detects **occupancy (motion)** using a series of inexpensive sensors (e.g. DHT11, PM2.5, infrared). It transforms raw sensor readings into **time-series windows** and uses a deep Conv1D neural network to classify whether the room is **occupied or empty**.

✅ Designed for **resource-constrained low-power devices, enabled with an operating system**
✅ Model trained and optimized in **TensorFlow**, converted to **TensorFlow Lite (TFLite)**  
✅ Evaluated with metrics suitable for deployment and academic research

---

## 🎯 Objectives

<p align="center">
  <img src="../system%20design%20occupancy%20monitoring%20TINYML.png" alt="System Overview" width="500">
</p>

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



<p align="center">
  <img src="../NN%20Architecture.png" alt="NN Architecture" width="500">
</p>



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
- `model_float32.tflite` – Full precision for testing - 32 bit
- `model_float8.tflite` – Optimized for deployment - 8 bit

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

The Pandas and Numpy library, are very imporant to carry out data cleaning/preprocessing tasks.

---

## ▶️ How to Run

1. Simply make sure your microcontroller/single board computer has a Linus OS, or a bare-metal Operating System that can run Machine Learning models, e.g. Tensorflow Lite Micro etc.

2. Attach the Dataset that you would like to train the model with

3. 

```bash
python occupancy_conv1d.py
```

Ensure the CSV file is in the correct path and named `merged_env_motion.csv`.

---

## 🔗 Related Work

- 📊 [Project Poster](../poster.pdf) 
- 💻 Runs on: Raspberry Pi 5 Model B (on-device inference tested)

---

## 🛠️ Deployment Notes

- Model supports conversion to TFLite (float16) for **TinyML applications**
- Ideal for integration with sensor platforms using **Python + TensorFlow Lite Runtime**
- May be extended to real-time inference with camera + sensor fusion
- May be extended to real-time inference with low-power systems, e.g. Raspberry Pi Nano microcontrollers
---

## 📚 Citation


```
Y. B. Perez Condori, "Edge AI Occupancy Detection Using Conv1D and Environmental Sensor Data," Final Year Project, Brunel University of London, 2025.
```

---

## 📄 License

MIT License – use, share, or extend freely.
