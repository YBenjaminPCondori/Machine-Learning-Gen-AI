
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Edge AI Occupancy Detection - Final Year Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            background-color: #fdfdfd;
            color: #333;
        }
        h1, h2 {
            color: #004aad;
        }
        code, pre {
            background-color: #f4f4f4;
            padding: 6px;
            border-radius: 4px;
            display: block;
            margin-bottom: 10px;
        }
        hr {
            border: none;
            border-top: 1px solid #ccc;
            margin: 40px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #bbb;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #ddd;
        }
        .center {
            text-align: center;
        }
        .highlight {
            color: #d6336c;
        }
    </style>
</head>
<body>

<h1 class="center">📡 Final Year Project – Edge AI Occupancy Detection using Conv1D</h1>
<p class="center"><em>This repository presents the complete pipeline for deploying an Edge AI occupancy detection model using environmental sensor data and a lightweight Conv1D neural network.</em></p>
<p class="center"><strong>BEng Computer Systems Engineering | Brunel University London | 2025</strong></p>

<hr/>

<h2>📌 Project Summary</h2>
<p>This system detects <strong>occupancy (motion)</strong> using inexpensive sensors (DHT11, PM2.5, IR) and processes the data into <strong>time-series windows</strong> for classification via a deep <span class="highlight">1D Convolutional Neural Network (Conv1D)</span>.</p>
<ul>
    <li>✅ Built for <strong>resource-constrained edge devices</strong></li>
    <li>✅ Trained using <strong>TensorFlow</strong>, optimized with <strong>TFLite</strong> for deployment</li>
    <li>✅ Evaluated with real-world ML metrics (F1, ROC-AUC, confusion matrix)</li>
</ul>

<h2>🎯 Objectives</h2>
<p class="center"><img src="../system%20design%20occupancy%20monitoring%20TINYML.png" alt="System Overview" width="600"/></p>
<ul>
    <li>Predict occupancy from multi-sensor time-series data</li>
    <li>Optimize for <strong>TinyML / Edge AI</strong> deployment</li>
    <li>Evaluate using F1 Score, ROC-AUC, and Confusion Matrix</li>
    <li>Ensure compatibility with low-power platforms like <strong>Raspberry Pi</strong></li>
</ul>

<h2>🗂️ Dataset Format</h2>
<p>File required: <code>merged_env_motion.csv</code></p>
<pre><code>timestamp, temperature, humidity, alcohol, pm2.5, motion
2024-01-01 00:00, 21.5, 45.2, 0.03, 12, 1
...</code></pre>
<ul>
    <li>🟢 Features: <code>temperature</code>, <code>humidity</code>, <code>alcohol</code>, <code>pm2.5</code></li>
    <li>🔵 Label: <code>motion</code> (binary: 0 = no motion, 1 = motion)</li>
    <li>⏱️ Timestamp used for window slicing (optional)</li>
</ul>

<h2>🧠 Model Architecture</h2>
<p class="center"><img src="../NN%20Architecture.png" alt="Conv1D Model Architecture" width="500"/></p>
<ul>
    <li>Loss Function: <strong>Binary Crossentropy</strong></li>
    <li>Optimizer: <strong>Adam</strong></li>
    <li>Training techniques: <strong>Class balancing</strong>, <strong>Early stopping</strong></li>
</ul>

<h2>📊 Evaluation Metrics</h2>
<ul>
    <li>✔️ Accuracy, Precision, Recall</li>
    <li>✔️ <strong>F1 Score</strong>, <strong>ROC-AUC</strong>, Log Loss</li>
    <li>✔️ <strong>Confusion Matrix</strong> (heatmap)</li>
    <li>✔️ <strong>ROC Curve</strong></li>
</ul>

<h2>💾 Model Outputs</h2>
<table>
    <tr><th>File</th><th>Format</th><th>Purpose</th></tr>
    <tr><td>best_conv1d_model.keras</td><td>Keras</td><td>Native model checkpoint</td></tr>
    <tr><td>model_float32.tflite</td><td>TFLite</td><td>High-precision testing</td></tr>
    <tr><td>model_float8.tflite</td><td>TFLite</td><td>Deployment-ready, 8-bit quantized</td></tr>
</table>

<h2>📦 Requirements</h2>
<pre><code>pip install pandas numpy scikit-learn tensorflow matplotlib seaborn</code></pre>
<p>📌 <em>Pandas and NumPy are essential for preprocessing and data handling.</em></p>

<h2>▶️ Running the Project</h2>
<ol>
    <li>Ensure you're using a Linux-based SBC (e.g. Raspberry Pi) or a microcontroller with support for ML inference.</li>
    <li>Place your dataset file as <code>merged_env_motion.csv</code> in the working directory.</li>
    <li>Run:</li>
</ol>
<pre><code>python occupancy_conv1d.py</code></pre>

<h2>🔗 Related Work</h2>
<ul>
    <li>📄 <a href="../poster.pdf">Project Poster</a></li>
    <li>🧪 Tested on: <strong>Raspberry Pi 5 Model B</strong> with on-device inference</li>
</ul>

<h2>🚀 Deployment Notes</h2>
<ul>
    <li>Compatible with <strong>TensorFlow Lite Micro</strong> and <strong>TFLite Runtime</strong></li>
    <li>Suitable for <strong>sensor fusion</strong>, <strong>real-time inference</strong>, and <strong>low-power ML</strong></li>
    <li>Expandable to include <strong>thermal imaging</strong>, <strong>infrared</strong>, or <strong>camera input</strong></li>
</ul>

<h2>📚 Citation</h2>
<pre><code>Y. B. Perez Condori, "Edge AI Occupancy Detection Using Conv1D and Environmental Sensor Data," Final Year Project, Brunel University of London, 2025.</code></pre>

<h2>📄 License</h2>
<p><strong>MIT License</strong> – Free to use, modify, and distribute.</p>

</body>
</html>
