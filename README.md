# 🔐 Attack Detection and Mitigation in Microgrid using AI/ML

> **SN Bose Internship Program, 2026**  
> Department of Electrical Engineering, NIT Silchar

---

## 📌 Project Overview

This project develops an intelligent **cyber-attack detection and mitigation system** for microgrids using Artificial Intelligence and Machine Learning techniques. Modern microgrids depend heavily on digital communication networks, making them vulnerable to sophisticated cyber-attacks. This system provides real-time detection and automated mitigation for three major attack types.

---

## ⚡ Attack Types Addressed

| Attack | Description |
|--------|-------------|
| **False Data Injection (FDI)** | Injects false sensor readings to mislead control systems |
| **Denial of Service (DoS)** | Overwhelms communication network causing system failure |
| **Replay Attack** | Retransmits old valid signals to deceive the control system |

---

## 🧠 ML Models & Accuracy

| Model | Accuracy |
|-------|----------|
| Random Forest (RF) | 79.00% |
| **Support Vector Machine (SVM)** | **82.25% ✅ Best** |
| Artificial Neural Network (ANN) | 81.62% |

---

## 🗂️ Project Structure

```
Microgrid_Project/
├── microgrid.py              # Main Python code
├── confusion_matrix.png      # Confusion Matrix Graph
├── voltage_comparison.png    # Voltage Signal Comparison
├── precision_scores.png      # Precision Score per Attack
├── model_comparison.png      # Model Accuracy Comparison
├── feature_importance.png    # Feature Importance (RF)
├── roc_curve.png             # ROC Curve with AUC Scores
├── fft_analysis.png          # FFT Frequency Domain Analysis
└── anomaly_detection.png     # Anomaly Detection Results
```

---

## 🔧 Methodology

### 1️⃣ Data Generation
- 4000 synthetic samples (1000 per class)
- Parameters: Voltage (V), Current (A), Frequency (Hz), Power (W)

### 2️⃣ Feature Extraction
- **Statistical:** Mean, Standard Deviation, Variance, Power Ratio, V-I Difference
- **Signal Processing:** Fast Fourier Transform (FFT) for frequency domain analysis

### 3️⃣ ML Classification
- Random Forest, SVM (RBF Kernel), ANN (MLP 64→32)
- 80% Train / 20% Test split

### 4️⃣ Anomaly Detection
- 3-Sigma statistical rule (Threshold = 3σ = 5.87V)
- Fast first-level screening before ML classification

### 5️⃣ Mitigation Strategies

| Attack | Immediate Action | Secondary Action |
|--------|-----------------|-----------------|
| FDI | Isolate compromised sensor | Switch to backup data source |
| DoS | Activate islanded mode | Redistribute loads |
| Replay | Reset communication channel | Activate backup controller |

---

## 📊 Key Results

- ✅ **SVM Best Accuracy: 82.25%**
- ✅ **FDI & DoS Detection: 100% Precision**
- ✅ **Anomaly Detection: FDI (9.18σ), DoS (33.23σ)**
- ✅ **8 Graphs Generated**
- ✅ **3 ML Models Trained & Compared**

---

## 🛠️ Installation & Usage

### Requirements
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run
```bash
python microgrid.py
```

---

## 🔮 Future Work

- Deep Learning: LSTM & CNN for temporal analysis
- Real dataset validation (IEEE 118-bus system)
- IoT sensor integration (MQTT/MODBUS)
- Blockchain-based security layer
- Edge deployment (NVIDIA Jetson / Raspberry Pi)
- Federated Learning for distributed microgrids
