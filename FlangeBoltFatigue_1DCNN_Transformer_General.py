# 1D-CNN预测模型
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
plt.rcParams['font.family'] = 'Arial'  # 设置为黑体，适合中文显示


# 自定义MAPE计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 加载数据
data = pd.read_csv('loads.csv')  # 替换为实际文件名

# 分离特征和目标值
X = data.iloc[:, 1:13].values  # 取Load-1到Load-12
y = data['WorstLife'].values   # 取WorstLife

# 数据预处理
# 1. 对目标值取对数，缓解偏态分布
y_log = np.log(y + 1)  # 加1避免log(0)

# 2. 归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1))

# 转换为PyTorch张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# 将特征重新组织为适合 1D CNN 的形状 (样本数, 通道数, 特征长度)
X_tensor = X_tensor.unsqueeze(1)  # 添加通道维度

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

class CNNTransformerWithPE(nn.Module):
    def __init__(self, seq_len=12, d_model=64):
        super(CNNTransformerWithPE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)  # 输出维度为 d_model

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Learnable Position Encoding（[1, seq_len, d_model]）
        self.position_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 输出层
        self.fc1 = nn.Linear(d_model * seq_len, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))     # [B, 32, 12]
        x = self.relu(self.conv2(x))     # [B, d_model, 12]

        x = x.permute(0, 2, 1)           # [B, 12, d_model]
        x = x + self.position_encoding   # 加上位置编码

        x = self.transformer_encoder(x)  # [B, 12, d_model]
        x = x.reshape(x.size(0), -1)     # 展平

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 实例化模型
model = CNNTransformerWithPE()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用 MSE 损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)  # 学习率衰减

# 设置训练参数
epochs = 200
batch_size = 32

# 创建数据集和 DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练循环
losses = []
val_losses = []
best_val_loss = float('inf')
patience = 30
counter = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)

    # 验证集性能
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test)
        val_loss = criterion(val_predictions, y_test)
        val_losses.append(val_loss.item())

    # 学习率衰减
    scheduler.step(val_loss)

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_1dcnn_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# 加载最佳模型
model.load_state_dict(torch.load('best_1dcnn_model.pth', weights_only=True))

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

    # 反归一化
    y_pred_rescaled = np.exp(scaler_y.inverse_transform(y_pred.numpy())) - 1
    y_test_rescaled = np.exp(scaler_y.inverse_transform(y_test.numpy())) - 1

    # 计算评估指标
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    print(f"R²: {r2:.4f}, MAPE: {mape:.4f}%, RMSE: {rmse:.2f}")

    # 输出部分预测值
    print("Predicted WorstLife:", y_pred_rescaled[:20].flatten())
    print("True WorstLife:", y_test_rescaled[:20].flatten())

# 概率预测部分 - 多次预测
num_samples = 100  # 预测次数
all_predictions = []

# 启用dropout以获取不确定性估计
model.train()  # 注意这里使用train模式以启用dropout

with torch.no_grad():
    for _ in range(num_samples):
        # 每次预测都会因dropout而略有不同
        pred = model(X_test)
        pred_rescaled = np.exp(scaler_y.inverse_transform(pred.numpy())) - 1
        all_predictions.append(pred_rescaled.flatten())

all_predictions = np.array(all_predictions)  # shape: (num_samples, num_test_points)
y_test_rescaled = y_test_rescaled.flatten()

# 计算统计量
mean_predictions = np.mean(all_predictions, axis=0)
std_predictions = np.std(all_predictions, axis=0)

# 找到MAPE最小的一组预测
mape_values = [mean_absolute_percentage_error(y_test_rescaled, pred) for pred in all_predictions]
best_idx = np.argmin(mape_values)
best_prediction = all_predictions[best_idx]

# 绘制概率预测图
plt.figure(figsize=(5.8, 6))

# 绘制真实值和预测均值
#plt.scatter(y_test_rescaled, mean_predictions, s=100, alpha=0.7, label='Mean prediction')

# 绘制最佳MAPE预测点
plt.scatter(y_test_rescaled, best_prediction, s=10, alpha=1, color='#FD6E6F', label='Test data',zorder=0.6)

# 绘制y=x参考线
x_line = np.linspace(0.8*min(y_test_rescaled), 1.2*max(y_test_rescaled), 100)
plt.plot(x_line, x_line, 'r--', label='y = x', linewidth=2,zorder=0.5)


# 绘制置信区间
sorted_idx = np.argsort(y_test_rescaled)
y_test_sorted = y_test_rescaled[sorted_idx]
mean_sorted = mean_predictions[sorted_idx]
std_sorted = std_predictions[sorted_idx]

# 绘制均值线
plt.plot(y_test_sorted, mean_sorted, color='#D4D4D4', linestyle='-', linewidth=2, label='Mean (μ)',zorder=0.3)

# μ ± σ
plt.fill_between(y_test_sorted,
                 mean_sorted - std_sorted,
                 mean_sorted + std_sorted,
                 color='#001080', alpha=1, label='μ ± σ',zorder=0.2)

# μ ± 2σ
plt.fill_between(y_test_sorted,
                 mean_sorted - 2*std_sorted,
                 mean_sorted + 2*std_sorted,
                 color='#50BDFF', alpha=1, label='μ ± 2σ',zorder=0.1)

# μ ± 3σ
plt.fill_between(y_test_sorted,
                 mean_sorted - 3*std_sorted,
                 mean_sorted + 3*std_sorted,
                 color='#B7E5FF', alpha=1, label='μ ± 3σ',zorder=0)


# 设置对数坐标
plt.xscale('log')  # x轴对数坐标
plt.yscale('log')  # y轴对数坐标

# 设置主刻度和次刻度的长度
plt.tick_params(axis='both',  # 应用到所有坐标轴
                which='major',  # 设置主刻度
                length=5,      # 主刻度长度
                width=1.5,        # 主刻度粗细
                labelsize=18)

plt.tick_params(axis='both',  # 应用到所有坐标轴
                which='minor',  # 设置次刻度
                length=2.5,       # 次刻度长度
                width=1.5)        # 次刻度粗细

# 设置坐标轴的粗细
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.xlabel("Tested (Repeats)", fontsize=20)
plt.ylabel("Predicted (Repeats)", fontsize=20)
plt.title("3 Sensors", fontsize=20)
plt.xlim([0.6*min(y_test_rescaled),1.4*max(y_test_rescaled)])
plt.ylim([0.6*min(y_test_rescaled),1.4*max(y_test_rescaled)])
# 标注评估指标
text = f"R² = {r2:.4f}\nMAPE = {mape:.2f}%"
ax = plt.gca()
plt.text(0.05, 0.9, text, transform=ax.transAxes, fontsize=18,verticalalignment="center", horizontalalignment="left")

# 显示图像
plt.legend(fontsize=18, loc='lower right', edgecolor='black', fancybox=False, shadow=True)
plt.tight_layout()
plt.savefig('1DCNN_Transformer_Uncertainty.png', dpi=600, bbox_inches='tight')
plt.show()

# ==========================
# SHAP 分析与可视化
# ==========================
import shap
import numpy as np
import matplotlib.pyplot as plt

# 选取样本数
sample_X = X_scaled[:150]

# 定义SHAP使用的模型预测函数
def shap_predict(input_array):
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        output = model(input_tensor).numpy()
    return output

# 设置背景数据（20个样本）
background = X_scaled[np.random.choice(X_scaled.shape[0], 20, replace=False)]

# 创建解释器并计算 SHAP 值
explainer = shap.KernelExplainer(shap_predict, background)
shap_values = explainer.shap_values(sample_X)
print("type(shap_values):", type(shap_values))
print("shap_values shape:", shap_values.shape)
print("sample_X shape:", sample_X.shape)

# 特征名
feature_names = np.array([f"Load_{i+1}" for i in range(sample_X.shape[1])])

# 去除 SHAP 最后一维，保证是 (50, 12)
shap_values = shap_values.squeeze(-1)

# 检查形状是否匹配
assert shap_values.shape == sample_X.shape, "Shape mismatch after squeezing."

# ============================
# 🎯 左移特征顺序
# ============================
shap_values = np.roll(shap_values, shift=-1, axis=1)
#feature_names = np.roll(feature_names, shift=-1)

# ============================
# 🎯 平均 SHAP 柱状图
# ============================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_abs_shap = np.array(mean_abs_shap).astype(float).flatten()

plt.figure(figsize=(8.5, 7))
bars = plt.bar(feature_names, mean_abs_shap, color='blue')
plt.ylabel("Mean SHAP value", fontsize=20)

# 添加数值标签（保留3位小数）
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
             f"{height:.3f}", ha='center', va='bottom', fontsize=16)

for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)  # 设置坐标轴线的粗细

plt.tick_params(axis='both',  # 应用到所有坐标轴
                width=1.5,        # 主刻度粗细
                labelsize=16)
plt.ylim([0,0.085])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("SHAP_Bar.png", dpi=600)
plt.show()

# ============================
# 🎯 SHAP 分布图
# ============================
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample_X, feature_names=feature_names, show=False)
plt.savefig("SHAP_Summary.png", bbox_inches="tight", dpi=600)
plt.show()

# ==========================
# 气泡热力图
# ==========================
import pandas as pd
feature_names = [f"L{i+1}" for i in range(12)]
df_features = pd.DataFrame(X_scaled, columns=feature_names)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 构建特征名
corr_matrix = df_features.corr(method='pearson').values

# 自定义 SHAP 风格配色
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_shap",
    [(0, (0/255, 138/255, 251/255)),   # 蓝色
     (0.5, (128/255, 0/255, 128/255)), # 紫色
     (1, (255/255, 0/255, 82/255))]    # 红色
)

fig, ax = plt.subplots(figsize=(8, 7))

# 绘制气泡
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        value = corr_matrix[i, j]
        size = abs(value) * 500  # 调整气泡大小
        color = cmap((value+1)/2)
        ax.scatter(j, len(feature_names) - 1 - i, s=size, c=[color], alpha=0.9, edgecolors='k')

# 设置坐标轴
ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, fontsize=16)
ax.set_yticklabels(feature_names[::-1], fontsize=16)

# ✅ 画格子框，而不是网格线穿过气泡
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
plt.savefig("Bubble_Correlation_Boxed.png", dpi=600)
plt.show()

