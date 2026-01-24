<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>
<h1>IoT Intrusion Detection Using Deep Learning</h1>

<p>A 1D Convolutional Neural Network implementation for detecting network intrusions in IoT environments, with emphasis on handling severe class imbalance and computational efficiency.</p>

<h2>🎯 Project Overview</h2>

<p>This project addresses the critical challenge of detecting rare cyberattacks in IoT networks where class imbalance severely impacts detection rates. Traditional models often achieve high overall accuracy while completely missing minority attack classes—a critical failure in cybersecurity applications.</p>

<p><strong>Key Achievement:</strong> Improved recall for minority attack classes from 0% to 43% by strategically trading 10% overall accuracy for significantly better rare attack detection.</p>

<h2>📊 Dataset</h2>

<ul>
    <li><strong>Source:</strong> IoT_Intrusion.csv (Kaggle)</li>
    <li><strong>Type:</strong> Network traffic patterns with multiple attack categories</li>
    <li><strong>Challenge:</strong> Severe class imbalance with rare attacks (Backdoor Malware, Browser Hijacking, XSS) having minimal representation</li>
</ul>

<p><strong>Attack Categories Detected:</strong></p>
<ul>
    <li>Backdoor Malware</li>
    <li>Benign Traffic</li>
    <li>Browser Hijacking</li>
    <li>Command Injection</li>
    <li>DDoS variants (ACK Fragmentation, HTTP Flood, ICMP Flood, etc.)</li>
    <li>SQL Injection</li>
    <li>XSS</li>
    <li>And more...</li>
</ul>

<h2>🏗️ Model Architecture</h2>

<h3>Lightweight 1D-CNN Design</h3>

<pre><code>Input Layer
↓
Conv1D(64 filters, kernel=5, padding=2) + BatchNorm + ReLU
↓
Conv1D(128 filters, kernel=3, padding=1) + BatchNorm + ReLU
↓
Conv1D(256 filters, kernel=3, padding=1) + BatchNorm + ReLU
↓
Global Average Pooling
↓
Fully Connected Layer
↓
Output (Multi-class Classification)</code></pre>
<p><strong>Design Principles:</strong></p>
<ul>
    <li>Minimal layers for computational efficiency</li>
    <li>Small kernel sizes (5, 3, 3) to reduce parameters</li>
    <li>Strategic padding to maintain spatial consistency</li>
    <li>Suitable for edge device deployment</li>
</ul>

<h2>🔬 Methodology</h2>

<h3>1. Data Preprocessing</h3>

<ul>
    <li><strong>Missing Value Handling:</strong> Mean imputation to preserve behavioral patterns</li>
    <li><strong>Feature Analysis:</strong> Frequency analysis and feature importance calculation</li>
    <li><strong>Normalization:</strong> Standard scaling for consistent feature ranges</li>
    <li><strong>Protocol Handling:</strong> Specialized processing for network protocol columns</li>
</ul>

<h3>2. Class Imbalance Solution</h3>

<ul>
    <li><strong>Problem:</strong> Rare attacks had 0% detection rate initially</li>
    <li><strong>Solution:</strong> Class-weighted cross-entropy loss with weights computed from inverse class frequencies</li>
    <li><strong>Impact:</strong> Eliminated zero-recall behavior for minority classes</li>
</ul>

<h3>3. Hyperparameter Optimization Journey</h3>

<p><strong>Initial Approach: Genetic Algorithms</strong></p>
<ul>
    <li>Explored evolutionary optimization</li>
    <li>Switched due to computational efficiency concerns</li>
</ul>

<p><strong>Final Approach: Bayesian Optimization</strong></p>
<ul>
    <li>More efficient hyperparameter search</li>
    <li>Guided by F1/Precision/Recall heatmaps</li>
    <li>Iterative refinement based on minority class performance</li>
</ul>

<p><strong>Optimized Hyperparameters:</strong></p>
<ul>
    <li>Learning Rate: 0.0001 (Adam optimizer)</li>
    <li>Batch Size: 4096 (optimized for 32GB RAM)</li>
    <li>Epochs: 50</li>
    <li>Scheduler: ReduceLROnPlateau</li>
    <li>Loss Function: Class-weighted Cross-Entropy</li>
</ul>

<h2>📈 Results</h2>

<h3>Performance Metrics</h3>

<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Final Model</th>
            <th>Change</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Accuracy</td>
            <td>91.45%</td>
            <td>81.45%</td>
            <td>-10.0%</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td>~0% (minority)</td>
            <td>62.45%</td>
            <td>+62.45%</td>
        </tr>
        <tr>
            <td>Precision</td>
            <td>-</td>
            <td>54.36%</td>
            <td>-</td>
        </tr>
        <tr>
            <td>F1-Score</td>
            <td>-</td>
            <td>54.68%</td>
            <td>-</td>
        </tr>
    </tbody>
</table>

<h3>Key Improvements by Attack Class</h3>

<table>
    <thead>
        <tr>
            <th>Attack Type</th>
            <th>Baseline Recall</th>
            <th>Final Recall</th>
            <th>Improvement</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Backdoor Malware</td>
            <td>0.00</td>
            <td>0.27</td>
            <td>+0.27</td>
        </tr>
        <tr>
            <td>Browser Hijacking</td>
            <td>0.00</td>
            <td>0.43</td>
            <td>+0.43</td>
        </tr>
        <tr>
            <td>Command Injection</td>
            <td>0.10</td>
            <td>0.14</td>
            <td>+0.04</td>
        </tr>
        <tr>
            <td>DDoS-HTTP Flood</td>
            <td>0.46</td>
            <td>0.80</td>
            <td>+0.34</td>
        </tr>
    </tbody>
</table>

<h3>The Strategic Trade-off</h3>

<p><strong>Decision:</strong> Sacrificed 10% overall accuracy to achieve 62% recall</p>

<p><strong>Rationale:</strong> In cybersecurity applications, missing a rare attack (false negative) is far more costly than a false alarm (false positive). This model prioritizes detection capability over raw accuracy.</p>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>

<pre><code>pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn</code></pre>
<h3>Installation</h3>

<pre><code>git clone https://github.com/YBenjaminPCondori/Machine-Learning-Gen-AI.git
cd Machine-Learning-Gen-AI/IoT\ Intrusion</code></pre>
<h3>Usage</h3>

<pre><code># Load and preprocess data
(See notebook for detailed preprocessing steps)
Train model
python train.py --epochs 50 --batch_size 4096 --lr 0.0001
Evaluate
python evaluate.py --model_path checkpoints/best_model.pth</code></pre>
<h2>📁 Repository Structure</h2>

<pre><code>IoT Intrusion/
├── notebooks/
│   └── IoT_Intrusion_Detection.ipynb
├── data/
│   └── IoT_Intrusion.csv
├── models/
│   └── cnn_model.py
├── utils/
│   ├── preprocessing.py
│   └── evaluation.py
├── results/
│   ├── confusion_matrices/
│   └── performance_plots/
└── README.md</code></pre>
<h2>🎓 Academic Context</h2>

<ul>
    <li><strong>Course:</strong> INM7062 - Programming and Mathematics for Artificial Intelligence</li>
    <li><strong>Institution:</strong> City, St George's University of London</li>
    <li><strong>Module Leader:</strong> Dr. Atif Riaz</li>
    <li><strong>Focus:</strong> Deep Learning Networks Using PyTorch (Task 2)</li>
</ul>

<h2>💡 Key Learnings</h2>

<ul>
    <li><strong>Class Imbalance is Critical:</strong> High accuracy can mask complete failure on minority classes</li>
    <li><strong>Optimization Matters:</strong> Choosing the right optimization strategy (Bayesian > Genetic for this case)</li>
    <li><strong>Domain Knowledge:</strong> In security applications, recall often matters more than accuracy</li>
    <li><strong>Efficiency by Design:</strong> Lightweight architectures enable edge deployment</li>
    <li><strong>Trade-offs are Strategic:</strong> Understanding when to sacrifice one metric for another</li>
</ul>

<h2>🔮 Future Work</h2>

<ul>
    <li><input type="checkbox"> Implement ensemble methods for improved minority class detection</li>
    <li><input type="checkbox"> Test on edge devices (Raspberry Pi, NVIDIA Jetson)</li>
    <li><input type="checkbox"> Explore attention mechanisms for interpretability</li>
    <li><input type="checkbox"> Real-time inference optimization</li>
    <li><input type="checkbox"> Transfer learning from similar network intrusion datasets</li>
</ul>

<h2>🤖 GenAI Use Statement</h2>

<p>Generative AI tools were used to assist with code commenting and documentation formatting. All model implementations, experiments, hyperparameter optimization, and results were independently designed, executed, and validated.</p>

<h2>📚 References</h2>

<ul>
    <li>Dataset: <a href="https://www.kaggle.com/">IoT Intrusion Dataset - Kaggle</a></li>
    <li>PyTorch Documentation: <a href="https://pytorch.org/docs/">https://pytorch.org/docs/</a></li>
    <li>Class Imbalance Techniques: Research papers on weighted loss functions</li>
</ul>

<h2>👨‍💻 Author</h2>

<p><strong>Yehoshua Benjamin Perez Condori</strong><br>
Student ID: 250057607<br>
GitHub: <a href="https://github.com/YBenjaminPCondori">@YBenjaminPCondori</a></p>

<h2>📄 License</h2>

<p>This project is part of academic coursework at City, St George's University of London.</p>

<hr>

<p>⭐ <strong>If you find this project interesting, please consider giving it a star!</strong></p>

<p><strong>Connect with me:</strong> <a href="https://linkedin.com">LinkedIn</a> | <a href="https://github.com/YBenjaminPCondori">GitHub</a></p>

<script>
    function copyMarkdown() {
        const markdown = `# IoT Intrusion Detection Using Deep Learning
A 1D Convolutional Neural Network implementation for detecting network intrusions in IoT environments, with emphasis on handling severe class imbalance and computational efficiency.
🎯 Project Overview
This project addresses the critical challenge of detecting rare cyberattacks in IoT networks where class imbalance severely impacts detection rates. Traditional models often achieve high overall accuracy while completely missing minority attack classes—a critical failure in cybersecurity applications.
Key Achievement: Improved recall for minority attack classes from 0% to 43% by strategically trading 10% overall accuracy for significantly better rare attack detection.
📊 Dataset

Source: IoT_Intrusion.csv (Kaggle)
Type: Network traffic patterns with multiple attack categories
Challenge: Severe class imbalance with rare attacks (Backdoor Malware, Browser Hijacking, XSS) having minimal representation

Attack Categories Detected:

Backdoor Malware
Benign Traffic
Browser Hijacking
Command Injection
DDoS variants (ACK Fragmentation, HTTP Flood, ICMP Flood, etc.)
SQL Injection
XSS
And more...

🏗️ Model Architecture
Lightweight 1D-CNN Design
```
Input Layer
↓
Conv1D(64 filters, kernel=5, padding=2) + BatchNorm + ReLU
↓
Conv1D(128 filters, kernel=3, padding=1) + BatchNorm + ReLU
↓
Conv1D(256 filters, kernel=3, padding=1) + BatchNorm + ReLU
↓
Global Average Pooling
↓
Fully Connected Layer
↓
Output (Multi-class Classification)
```
Design Principles:

Minimal layers for computational efficiency
Small kernel sizes (5, 3, 3) to reduce parameters
Strategic padding to maintain spatial consistency
Suitable for edge device deployment

🔬 Methodology
1. Data Preprocessing

Missing Value Handling: Mean imputation to preserve behavioral patterns
Feature Analysis: Frequency analysis and feature importance calculation
Normalization: Standard scaling for consistent feature ranges
Protocol Handling: Specialized processing for network protocol columns

2. Class Imbalance Solution

Problem: Rare attacks had 0% detection rate initially
Solution: Class-weighted cross-entropy loss with weights computed from inverse class frequencies
Impact: Eliminated zero-recall behavior for minority classes

3. Hyperparameter Optimization Journey
Initial Approach: Genetic Algorithms

Explored evolutionary optimization
Switched due to computational efficiency concerns

Final Approach: Bayesian Optimization

More efficient hyperparameter search
Guided by F1/Precision/Recall heatmaps
Iterative refinement based on minority class performance

Optimized Hyperparameters:

Learning Rate: 0.0001 (Adam optimizer)
Batch Size: 4096 (optimized for 32GB RAM)
Epochs: 50
Scheduler: ReduceLROnPlateau
Loss Function: Class-weighted Cross-Entropy

📈 Results
Performance Metrics
MetricBaselineFinal ModelChangeAccuracy91.45%81.45%-10.0%Recall~0% (minority)62.45%+62.45%Precision-54.36%-F1-Score-54.68%-
Key Improvements by Attack Class
Attack TypeBaseline RecallFinal RecallImprovementBackdoor Malware0.000.27+0.27Browser Hijacking0.000.43+0.43Command Injection0.100.14+0.04DDoS-HTTP Flood0.460.80+0.34
The Strategic Trade-off
Decision: Sacrificed 10% overall accuracy to achieve 62% recall
Rationale: In cybersecurity applications, missing a rare attack (false negative) is far more costly than a false alarm (false positive). This model prioritizes detection capability over raw accuracy.
🚀 Getting Started
Prerequisites
```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
```
Installation
```bash
git clone https://github.com/YBenjaminPCondori/Machine-Learning-Gen-AI.git
cd Machine-Learning-Gen-AI/IoT\ Intrusion
```
Usage
```python
Load and preprocess data
(See notebook for detailed preprocessing steps)
Train model
python train.py --epochs 50 --batch_size 4096 --lr 0.0001
Evaluate
python evaluate.py --model_path checkpoints/best_model.pth
```
📁 Repository Structure
```
IoT Intrusion/
├── notebooks/
│   └── IoT_Intrusion_Detection.ipynb
├── data/
│   └── IoT_Intrusion.csv
├── models/
│   └── cnn_model.py
├── utils/
│   ├── preprocessing.py
│   └── evaluation.py
├── results/
│   ├── confusion_matrices/
│   └── performance_plots/
└── README.md
```
🎓 Academic Context

Course: INM7062 - Programming and Mathematics for Artificial Intelligence
Institution: City, St George's University of London
Module Leader: Dr. Atif Riaz
Focus: Deep Learning Networks Using PyTorch (Task 2)

💡 Key Learnings

Class Imbalance is Critical: High accuracy can mask complete failure on minority classes
Optimization Matters: Choosing the right optimization strategy (Bayesian > Genetic for this case)
Domain Knowledge: In security applications, recall often matters more than accuracy
Efficiency by Design: Lightweight architectures enable edge deployment
Trade-offs are Strategic: Understanding when to sacrifice one metric for another

🔮 Future Work

 Implement ensemble methods for improved minority class detection
 Test on edge devices (Raspberry Pi, NVIDIA Jetson)
 Explore attention mechanisms for interpretability
 Real-time inference optimization
 Transfer learning from similar network intrusion datasets

🤖 GenAI Use Statement
Generative AI tools were used to assist with code commenting and documentation formatting. All model implementations, experiments, hyperparameter optimization, and results were independently designed, executed, and validated.

📚 References

Dataset: IoT Intrusion Dataset - Kaggle
PyTorch Documentation: https://pytorch.org/docs/
Class Imbalance Techniques: Research papers on weighted loss functions

👨‍💻 Author
Yehoshua Benjamin Perez Condori
GitHub: @YBenjaminPCondori

📄 License
This project is part of academic coursework at City, St George's University of London.


