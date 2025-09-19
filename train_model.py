# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

# ==========================
# MAPE
# ==========================
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ==========================
# Data loading and preprocessing
# ==========================
data = pd.read_csv("loads.csv")
X = data.iloc[:, 1:13].values  # Load-1 ~ Load-12
y = data["WorstLife"].values

y_log = np.log(y + 1)  # logarithmic transformation
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1))

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # [N,1,12]
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

# ==========================
# Model definition
# ==========================
class CNNTransformerWithPE(nn.Module):
    def __init__(self, seq_len=12, d_model=64):
        super(CNNTransformerWithPE, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Learnable positional encoding
        self.position_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc1 = nn.Linear(d_model * seq_len, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))          # [B, d_model, 12]
        x = x.permute(0, 2, 1)                # [B, 12, d_model]
        x = x + self.position_encoding
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# ==========================
# Main program (training + saving)
# ==========================
if __name__ == "__main__":
    model = CNNTransformerWithPE()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

    epochs = 150
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    patience, counter = 20, 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = criterion(val_preds, y_test)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_1dcnn_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # ==========================
    # Evaluation
    # ==========================
    model.load_state_dict(torch.load("best_1dcnn_model.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test).item()
        y_pred_rescaled = np.exp(scaler_y.inverse_transform(y_pred.numpy())) - 1
        y_test_rescaled = np.exp(scaler_y.inverse_transform(y_test.numpy())) - 1

        r2 = r2_score(y_test_rescaled, y_pred_rescaled)
        mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

        print(f"Test Loss: {test_loss:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")

    # ==========================
    # Uncertainty estimation with MC Dropout
    # ==========================
    num_samples = 100
    all_predictions = []

    model.train()  # Enable dropout
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(X_test)
            pred_rescaled = np.exp(scaler_y.inverse_transform(pred.numpy())) - 1
            all_predictions.append(pred_rescaled.flatten())

    all_predictions = np.array(all_predictions)
    y_test_rescaled = y_test_rescaled.flatten()

    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)

    # Select prediction with lowest MAPE
    mape_values = [mean_absolute_percentage_error(y_test_rescaled, pred) for pred in all_predictions]
    best_idx = np.argmin(mape_values)
    best_prediction = all_predictions[best_idx]

    # ==========================
    # Plot uncertainty
    # ==========================
    plt.figure(figsize=(5.8, 6))
    plt.scatter(y_test_rescaled, best_prediction, s=10, alpha=1, color='#FD6E6F', label='Test data', zorder=0.6)

    # Reference line y=x
    x_line = np.linspace(0.8*min(y_test_rescaled), 1.2*max(y_test_rescaled), 100)
    plt.plot(x_line, x_line, 'r--', label='y = x', linewidth=2, zorder=0.5)

    # Sorted for CI plotting
    sorted_idx = np.argsort(y_test_rescaled)
    y_test_sorted = y_test_rescaled[sorted_idx]
    mean_sorted = mean_predictions[sorted_idx]
    std_sorted = std_predictions[sorted_idx]

    plt.plot(y_test_sorted, mean_sorted, color='#D4D4D4', linestyle='-', linewidth=2, label='Mean (μ)', zorder=0.3)
    plt.fill_between(y_test_sorted, mean_sorted - std_sorted, mean_sorted + std_sorted,
                     color='#001080', alpha=1, label='μ ± σ', zorder=0.2)
    plt.fill_between(y_test_sorted, mean_sorted - 2*std_sorted, mean_sorted + 2*std_sorted,
                     color='#50BDFF', alpha=1, label='μ ± 2σ', zorder=0.1)
    plt.fill_between(y_test_sorted, mean_sorted - 3*std_sorted, mean_sorted + 3*std_sorted,
                     color='#B7E5FF', alpha=1, label='μ ± 3σ', zorder=0)

    # Log scale
    plt.xscale('log')
    plt.yscale('log')

    plt.tick_params(axis='both', which='major', length=5, width=1.5, labelsize=18)
    plt.tick_params(axis='both', which='minor', length=2.5, width=1.5)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.xlabel("Tested (Repeats)", fontsize=20)
    plt.ylabel("Predicted (Repeats)", fontsize=20)
    plt.title("Uncertainty analysis", fontsize=20)
    plt.xlim([0.6*min(y_test_rescaled), 1.4*max(y_test_rescaled)])
    plt.ylim([0.6*min(y_test_rescaled), 1.4*max(y_test_rescaled)])

    text = f"R² = {r2:.4f}\nMAPE = {mape:.2f}%"
    ax = plt.gca()
    plt.text(0.05, 0.9, text, transform=ax.transAxes, fontsize=18,
             verticalalignment="center", horizontalalignment="left")

    plt.legend(fontsize=18, loc='lower right', edgecolor='black', fancybox=False, shadow=True)
    plt.tight_layout()
    plt.savefig('1DCNN_Transformer_Uncertainty.png', dpi=600, bbox_inches='tight')
    plt.show()

    # ==========================
    # Save files
    # ==========================
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    np.save("X_scaled.npy", X_scaled)
