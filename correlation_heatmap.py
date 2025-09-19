import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 加载归一化后的特征
X_scaled = np.load("X_scaled.npy")

# 构建 DataFrame
feature_names = [f"L{i+1}" for i in range(12)]
df_features = pd.DataFrame(X_scaled, columns=feature_names)

# 计算相关性矩阵
corr_matrix = df_features.corr(method="pearson").values

# 自定义 SHAP 风格配色
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_shap",
    [(0, (0/255, 138/255, 251/255)),   # 蓝色
     (0.5, (128/255, 0/255, 128/255)), # 紫色
     (1, (255/255, 0/255, 82/255))]    # 红色
)

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制气泡
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        value = corr_matrix[i, j]
        size = abs(value) * 500
        color = cmap((value + 1) / 2)
        ax.scatter(j, len(feature_names) - 1 - i, s=size, c=[color], alpha=0.9, edgecolors="k")

# 设置坐标轴
ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, fontsize=16)
ax.set_yticklabels(feature_names[::-1], fontsize=16)

# 格子框
ax.set_xticks(np.arange(-0.5, len(feature_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_xlim(-0.5, len(feature_names)-0.5)
ax.set_ylim(-0.5, len(feature_names)-0.5)

# 添加颜色条
norm = mcolors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Pearson Correlation", fontsize=16)

ax.set_title("Bubble heat map of bolt preload correlations", fontsize=16)
plt.tight_layout()
plt.savefig("Bubble_Correlation.png", dpi=600)
plt.show()
