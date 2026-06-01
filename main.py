import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ====================================
# Create Long Sequential Dataset
# ====================================

sequence_data = []
targets = []

for i in range(1, 101):
    seq = [i + j for j in range(20)]  # length 20
    sequence_data.append(seq)
    targets.append(i + 20)

# ====================================
# Normalize Data
# ====================================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(np.array(sequence_data).reshape(-1, 1)).reshape(len(sequence_data), 20, 1)
y_scaled = scaler_y.fit_transform(np.array(targets).reshape(-1, 1))

X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(y_scaled, dtype=torch.float32)

print("Input Shape:", X.shape)
print("Target Shape:", y.shape)

# ====================================
# RNN Model
# ====================================

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, hidden = self.rnn(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

# ====================================
# LSTM Model
# ====================================

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

# ====================================
# Training Function
# ====================================

def train_model(model, X, y, name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"{name} | Epoch {epoch+1} | Loss: {loss.item():.4f}")

    return model

# ====================================
# Train RNN
# ====================================

print("\nTraining RNN...\n")
rnn_model = RNNModel()
train_model(rnn_model, X, y, "RNN")

# ====================================
# Train LSTM
# ====================================

print("\nTraining LSTM...\n")
lstm_model = LSTMModel()
train_model(lstm_model, X, y, "LSTM")

# ====================================
# Test Sequence
# ====================================

test_seq = [101 + i for i in range(20)]
test_scaled = scaler_X.transform(np.array(test_seq).reshape(-1, 1)).reshape(1, 20, 1)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

# ====================================
# Predictions
# ====================================

with torch.no_grad():
    rnn_pred_scaled = rnn_model(test_tensor).numpy()
    lstm_pred_scaled = lstm_model(test_tensor).numpy()

# Inverse transform to original scale
rnn_pred = scaler_y.inverse_transform(rnn_pred_scaled)[0][0]
lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0][0]

print("\nExpected Value:")
print(121)

print("\nRNN Prediction:")
print(rnn_pred)

print("\nLSTM Prediction:")
print(lstm_pred)
