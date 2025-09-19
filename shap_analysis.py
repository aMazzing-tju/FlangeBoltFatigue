# shap_analysis.py
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
from train_model import CNNTransformerWithPE

# ======================
# Load model & data
# ======================
scaler_X = joblib.load("scaler_X.pkl")
X_scaled = np.load("X_scaled.npy")

model = CNNTransformerWithPE()
model.load_state_dict(torch.load("best_1dcnn_model.pth", weights_only=True))
model.eval()

# ======================
# SHAP analysis
# ======================
sample_X = X_scaled[:150]

def shap_predict(input_array):
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        output = model(input_tensor).numpy()
    return output

# Background dataset for KernelSHAP
background = X_scaled[np.random.choice(X_scaled.shape[0], 20, replace=False)]
explainer = shap.KernelExplainer(shap_predict, background)
shap_values = explainer.shap_values(sample_X)

# Feature names and preprocessing
feature_names = np.array([f"Load_{i+1}" for i in range(sample_X.shape[1])])
shap_values = np.array(shap_values).squeeze(-1)
shap_values = np.roll(shap_values, shift=-1, axis=1)  # Shift feature order left

# ======================
# Mean SHAP bar plot
# ======================
mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()

plt.figure(figsize=(9, 7))
bars = plt.bar(feature_names, mean_abs_shap, color="blue")
plt.ylabel("Mean SHAP value", fontsize=20)

# Add numeric labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
             f"{height:.3f}", ha="center", va="bottom", fontsize=16)

plt.tick_params(axis="both", width=1.5, labelsize=16)
plt.ylim([0, 0.1])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("SHAP_Bar.png", dpi=600)
plt.show()

# ======================
# SHAP summary plot
# ======================
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample_X, feature_names=feature_names, show=False)
plt.savefig("SHAP_Summary.png", bbox_inches="tight", dpi=600)
plt.show()
