"""Архитектуры RNN, LSTM и GRU."""

import torch
from torch import nn
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)

        last_output = rnn_output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.fc(last_output).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.fc(last_output).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_output, _ = self.gru(x)

        last_output = gru_output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.fc(last_output).squeeze(-1)
