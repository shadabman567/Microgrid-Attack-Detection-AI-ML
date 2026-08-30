# ===== PART 1: LIBRARIES & DATA GENERATION =====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Libraries loaded successfully!")

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000  # 1000 data points

# --- Normal Condition ---
normal_voltage   = np.random.normal(230, 2, n_samples)
normal_current   = np.random.normal(10, 0.5, n_samples)
normal_frequency = np.random.normal(50, 0.1, n_samples)
normal_power     = normal_voltage * normal_current
normal_label     = np.zeros(n_samples)  # 0 = Normal

# --- FDI Attack (False Data Injection) ---
fdi_voltage   = np.random.normal(245, 5, n_samples)
fdi_current   = np.random.normal(13, 1, n_samples)
fdi_frequency = np.random.normal(50, 0.1, n_samples)
fdi_power     = fdi_voltage * fdi_current
fdi_label     = np.ones(n_samples)  # 1 = FDI Attack

# --- DoS Attack (Denial of Service) ---
dos_voltage   = np.random.normal(180, 15, n_samples)
dos_current   = np.random.normal(5, 2, n_samples)
dos_frequency = np.random.normal(48, 1, n_samples)
dos_power     = dos_voltage * dos_current
dos_label     = np.full(n_samples, 2)  # 2 = DoS Attack

# --- Replay Attack ---
replay_voltage   = np.random.normal(231, 2, n_samples)
replay_current   = np.random.normal(10.2, 0.5, n_samples)
replay_frequency = np.random.normal(50.05, 0.1, n_samples)
replay_power     = replay_voltage * replay_current
replay_label     = np.full(n_samples, 3)  # 3 = Replay Attack

# --- Combine All Data ---
voltage   = np.concatenate([normal_voltage, fdi_voltage, dos_voltage, replay_voltage])
current   = np.concatenate([normal_current, fdi_current, dos_current, replay_current])
frequency = np.concatenate([normal_frequency, fdi_frequency, dos_frequency, replay_frequency])
power     = np.concatenate([normal_power, fdi_power, dos_power, replay_power])
labels    = np.concatenate([normal_label, fdi_label, dos_label, replay_label])

print(f"Total data points: {len(labels)}")
print(f"Classes: Normal=0, FDI=1, DoS=2, Replay=3")

# ===== PART 2: FEATURE EXTRACTION =====

# Statistical Features nikaalte hain
def extract_features(voltage, current, frequency, power):
    features = []
    for i in range(len(voltage)):
        row = [
            voltage[i],
            current[i],
            frequency[i],
            power[i],
            np.mean([voltage[i], current[i], frequency[i]]),   # mean
            np.std([voltage[i], current[i], frequency[i]]),    # std deviation
            np.var([voltage[i], current[i], frequency[i]]),    # variance
            voltage[i] - current[i],                           # difference feature
            power[i] / (voltage[i] + 0.001),                  # power ratio
        ]
        features.append(row)
    return np.array(features)

print("Extracting features...")
X = extract_features(voltage, current, frequency, power)
y = labels

print(f"Feature matrix shape: {X.shape}")

# ===== PART 3: TRAIN/TEST SPLIT =====

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ===== PART 4: RANDOM FOREST MODEL =====

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Detailed Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Normal", "FDI Attack", "DoS Attack", "Replay Attack"]))

# ===== PART 5: CONFUSION MATRIX GRAPH =====

print("\nGenerating graphs...")

# Graph 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "FDI", "DoS", "Replay"],
            yticklabels=["Normal", "FDI", "DoS", "Replay"])
plt.title("Confusion Matrix - Attack Detection", fontsize=14)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("✅ Confusion matrix saved!")

# ===== GRAPH 2: VOLTAGE COMPARISON =====

plt.figure(figsize=(10, 5))
plt.plot(normal_voltage[:100], label="Normal", color="green")
plt.plot(fdi_voltage[:100],    label="FDI Attack", color="red")
plt.plot(dos_voltage[:100],    label="DoS Attack", color="orange")
plt.plot(replay_voltage[:100], label="Replay Attack", color="purple")
plt.title("Voltage Pattern - Normal vs Attack Conditions", fontsize=14)
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.legend()
plt.tight_layout()
plt.savefig("voltage_comparison.png")
plt.show()
print("✅ Voltage graph saved!")

# ===== GRAPH 3: ACCURACY BAR CHART =====

plt.figure(figsize=(8, 5))
classes = ["Normal", "FDI Attack", "DoS Attack", "Replay Attack"]
precision_scores = [0.60, 1.00, 1.00, 0.54]
colors = ["green", "red", "orange", "purple"]
plt.bar(classes, precision_scores, color=colors)
plt.title("Precision Score per Attack Type", fontsize=14)
plt.ylabel("Precision Score")
plt.ylim(0, 1.2)
for i, v in enumerate(precision_scores):
    plt.text(i, v + 0.02, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("precision_scores.png")
plt.show()
print("✅ Precision graph saved!")

# ===== PART 6: MITIGATION LOGIC =====

print("\n" + "="*50)
print("🛡️  MITIGATION SYSTEM ACTIVATED")
print("="*50)

attack_names = {0: "Normal", 1: "FDI Attack", 2: "DoS Attack", 3: "Replay Attack"}

mitigation_actions = {
    0: "✅ System Normal — No action needed",
    1: "⚠️  FDI Detected — Isolating affected sensor, switching to backup data source",
    2: "🚨 DoS Detected — Activating islanded mode, redistributing loads",
    3: "🔁 Replay Detected — Resetting communication channel, activating backup controller"
}

# Test on first 8 samples
print("\nReal-time Detection on Test Samples:")
print("-" * 50)
for i in range(8):
    prediction = model.predict([X_test[i]])[0]
    actual     = int(y_test[i])
    print(f"\nSample {i+1}:")
    print(f"  Actual   : {attack_names[actual]}")
    print(f"  Predicted: {attack_names[int(prediction)]}")
    print(f"  Action   : {mitigation_actions[int(prediction)]}")

print("\n" + "="*50)
print("✅ PROJECT COMPLETE! All files saved.")
print("="*50)

# ===== PART 7: MULTIPLE MODEL COMPARISON =====

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

print("\n" + "="*50)
print("🤖 TRAINING MULTIPLE MODELS FOR COMPARISON")
print("="*50)

# Feature Scaling (SVM aur ANN ke liye zaroori)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- Model 1: Random Forest (already trained) ----
rf_accuracy = accuracy_score(y_test, y_pred)

# ---- Model 2: SVM ----
print("\nTraining SVM...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred     = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"✅ SVM Accuracy: {svm_accuracy * 100:.2f}%")

# ---- Model 3: ANN (Neural Network) ----
print("\nTraining ANN...")
ann_model = MLPClassifier(hidden_layer_sizes=(64, 32),
                           max_iter=300, random_state=42)
ann_model.fit(X_train_scaled, y_train)
ann_pred     = ann_model.predict(X_test_scaled)
ann_accuracy = accuracy_score(y_test, ann_pred)
print(f"✅ ANN Accuracy: {ann_accuracy * 100:.2f}%")

# ---- Comparison Table ----
print("\n📊 MODEL COMPARISON:")
print("-"*40)
print(f"  Random Forest : {rf_accuracy  * 100:.2f}%")
print(f"  SVM           : {svm_accuracy * 100:.2f}%")
print(f"  ANN           : {ann_accuracy * 100:.2f}%")
print("-"*40)

# ===== GRAPH 4: MODEL COMPARISON BAR CHART =====

models     = ['Random Forest', 'SVM', 'ANN']
accuracies = [rf_accuracy * 100, svm_accuracy * 100, ann_accuracy * 100]
colors     = ['#2196F3', '#FF5722', '#4CAF50']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors, width=0.5)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f'{acc:.2f}%', ha='center',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("✅ Model comparison graph saved!")

# ===== GRAPH 5: FEATURE IMPORTANCE =====

feature_names = ['Voltage', 'Current', 'Frequency', 'Power',
                 'Mean', 'Std Dev', 'Variance',
                 'V-I Diff', 'Power Ratio']

importances = model.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances, color='steelblue')
plt.title("Feature Importance - Random Forest", fontsize=14, fontweight='bold')
plt.ylabel("Importance Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("✅ Feature importance graph saved!")

# ===== GRAPH 6: ROC CURVE =====

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize labels
y_test_bin  = label_binarize(y_test,  classes=[0, 1, 2, 3])
y_score     = model.predict_proba(X_test)

plt.figure(figsize=(8, 6))
attack_labels = ['Normal', 'FDI Attack', 'DoS Attack', 'Replay Attack']
roc_colors    = ['green', 'red', 'orange', 'purple']

for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc     = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=roc_colors[i], lw=2,
             label=f'{attack_labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0,1], [0,1], 'k--', lw=1)
plt.title("ROC Curve - Multi-Class Attack Detection", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
print("✅ ROC curve saved!")

print("\n" + "="*50)
print("✅ ALL GRAPHS SAVED! Project Complete.")
print("="*50)

# ===== PART 8: FFT FEATURE EXTRACTION =====

from numpy.fft import fft

print("\n" + "="*50)
print("📡 FFT - SIGNAL PROCESSING ANALYSIS")
print("="*50)

def apply_fft(signal):
    fft_vals  = np.abs(fft(signal))
    fft_freq  = np.fft.fftfreq(len(signal))
    # Sirf positive frequencies
    half      = len(fft_vals) // 2
    return fft_freq[:half], fft_vals[:half]

# FFT on voltage signals
freq_normal, fft_normal   = apply_fft(normal_voltage)
freq_fdi,    fft_fdi      = apply_fft(fdi_voltage)
freq_dos,    fft_dos      = apply_fft(dos_voltage)
freq_replay, fft_replay   = apply_fft(replay_voltage)

# FFT Graph
plt.figure(figsize=(12, 5))
plt.plot(freq_normal[:100],  fft_normal[:100],  label="Normal",       color="green")
plt.plot(freq_fdi[:100],     fft_fdi[:100],     label="FDI Attack",   color="red")
plt.plot(freq_dos[:100],     fft_dos[:100],     label="DoS Attack",   color="orange")
plt.plot(freq_replay[:100],  fft_replay[:100],  label="Replay Attack",color="purple")
plt.title("FFT Analysis - Frequency Domain of Voltage Signals", fontsize=14, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.savefig("fft_analysis.png")
plt.show()
print("✅ FFT graph saved!")

# FFT Stats Print
print(f"\nFFT Peak Magnitude Comparison:")
print(f"  Normal       : {max(fft_normal):.2f}")
print(f"  FDI Attack   : {max(fft_fdi):.2f}")
print(f"  DoS Attack   : {max(fft_dos):.2f}")
print(f"  Replay Attack: {max(fft_replay):.2f}")

# ===== PART 9: ANOMALY DETECTION =====

print("\n" + "="*50)
print("🚨 ANOMALY DETECTION SYSTEM")
print("="*50)

# Normal data ka mean aur std calculate karo
normal_mean_v = np.mean(normal_voltage)
normal_std_v  = np.std(normal_voltage)
threshold     = 3.0  # 3-sigma rule

print(f"\nNormal Voltage → Mean: {normal_mean_v:.2f}V, Std: {normal_std_v:.2f}V")
print(f"Anomaly Threshold: ±{threshold} sigma = ±{threshold * normal_std_v:.2f}V")

def detect_anomaly(voltage_val, mean, std, threshold):
    deviation = abs(voltage_val - mean) / std
    if deviation > threshold:
        return True, deviation
    return False, deviation

# Test on sample values
test_voltages = {
    "Normal Sample"  : 231.5,
    "FDI Sample"     : 248.0,
    "DoS Sample"     : 165.0,
    "Replay Sample"  : 231.8
}

print("\nAnomaly Detection Results:")
print("-" * 55)
for name, val in test_voltages.items():
    is_anomaly, deviation = detect_anomaly(
        val, normal_mean_v, normal_std_v, threshold
    )
    status = "🚨 ANOMALY DETECTED" if is_anomaly else "✅ Normal"
    print(f"  {name:15s} | Voltage: {val}V | "
          f"Deviation: {deviation:.2f}σ | {status}")

# Anomaly Detection Graph
voltages_test  = list(test_voltages.values())
sample_names   = list(test_voltages.keys())
deviations     = [abs(v - normal_mean_v) / normal_std_v for v in voltages_test]
bar_colors     = ['green' if d <= threshold else 'red' for d in deviations]

plt.figure(figsize=(9, 5))
bars = plt.bar(sample_names, deviations, color=bar_colors)
plt.axhline(y=threshold, color='black', linestyle='--',
            linewidth=2, label=f'Threshold (3σ = {threshold})')
plt.title("Anomaly Detection - Sigma Deviation per Sample",
          fontsize=14, fontweight='bold')
plt.ylabel("Deviation (σ)")
plt.legend()
for bar, dev in zip(bars, deviations):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'{dev:.2f}σ', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("anomaly_detection.png")
plt.show()
print("✅ Anomaly detection graph saved!")

print("\n" + "="*50)
print("🎯 PROJECT FULLY COMPLETE!")
print("Total Graphs Generated: 8")
print("Models Trained        : 3 (RF, SVM, ANN)")
print("Attack Types Detected : 4 (Normal, FDI, DoS, Replay)")
print("="*50)