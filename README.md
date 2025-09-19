# FlangeBoltFatigue

This repository contains the source codes and processed dataset for the paper:

**"From Contact to Prognosis: A Self-Powered and Explainable Framework for Fatigue Monitoring of Wind Turbine Joints"**

It provides a reproducible implementation of a **1D-CNN + Transformer model** for fatigue life prediction of flange bolts, together with **SHAP-based interpretability analysis** and a **feature correlation heat map**.

---

## 📂 Repository Structure

```
project/
  ├── train_model.py             # Model training and evaluation
  ├── shap_analysis.py           # SHAP-based feature importance analysis
  ├── correlation_heatmap.py     # Feature correlation bubble heatmap
  ├── loads.csv                  # Example dataset
  ├── best_1dcnn_model.pth       # Trained model weights (generated after training)
  ├── scaler_X.pkl               # Saved feature scaler
  ├── scaler_y.pkl               # Saved target scaler
  ├── X_scaled.npy               # Normalized feature matrix
  └── requirements.txt           # Dependencies
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/aMazzing-tju/FlangeBoltFatigue.git
cd FlangeBoltFatigue
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model
```bash
python train_model.py
```
This will:
- Train the CNN–Transformer model
- Save the best model as `best_1dcnn_model.pth`
- Save scalers (`scaler_X.pkl`, `scaler_y.pkl`)
- Save processed features (`X_scaled.npy`)

### 2. Run SHAP analysis
```bash
python shap_analysis.py
```
This will:
- Load the trained model
- Compute SHAP values for 150 test samples
- Generate:
  - `SHAP_Bar.png` (mean absolute SHAP values)
  - `SHAP_Summary.png` (SHAP summary plot)

### 3. Generate correlation heatmap
```bash
python correlation_heatmap.py
```
This will:
- Compute Pearson correlations among bolt loads
- Generate `Bubble_Correlation.png` (bubble heatmap)
