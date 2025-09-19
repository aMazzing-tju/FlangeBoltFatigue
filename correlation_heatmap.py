# correlation_heatmap.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ======================
# Load preprocessed features
# ======================
X_scaled = np.load("X_scaled.npy")
feature_names = [f"L{i+1}" for i in range(12)]
df_features = pd.DataFrame(X_scaled, columns=feature_names)

# ======================
# Compute Pearson correlation matrix
# ======================
corr_matrix = df_features.corr(method="pearson").values

# ======================
# Define SHAP-style custom colormap
# ======================
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_shap",
    [
        (0, (0/255, 138/255, 251/255)),   # Blue
        (0.5, (128/255, 0/255, 128/255)), # Purple
        (1, (255/255, 0/255, 82/255))     # Red
    ]
)

# ======================
# Plot bubble heatmap
# ======================
fig, ax = plt.subplots(figsize=(8, 7))

# Draw bubbles
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        value = corr_matrix[i, j]
        size = abs(value) * 500  # Bubble size proportional to |correlation|
        color = cmap((value+1)/2)
        ax.scatter(j, len(feature_names) - 1 - i, s=size, c=[color],
                   alpha=0.9, edgecolors="k")

# Configure axes
ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, fontsize=16)
ax.set_yticklabels(feature_names[::-1], fontsize=16)

# Draw grid boxes around bubbles
ax.set_xticks(np.arange(-0.5, len(feature_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_xlim(-0.5, len(feature_names)-0.5)
ax.set_ylim(-0.5, len(feature_names)-0.5)

# Add colorbar
norm = mcolors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.15, aspect=40)
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Pearson Correlation", fontsize=16)

# Title
ax.set_title("Bubble heatmap of bolt preload correlations", fontsize=16)

plt.tight_layout()
plt.savefig("Bubble_Correlation.png", dpi=600)
plt.show()
