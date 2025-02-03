import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # x -> batch_size, seq_length, input_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length, hidden_size)

        # hidden layer that's repeatedly fed into the RNN layers
        # h0: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)
        # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)
        out = self.linear(out)
        # out: (batch_size, output_size)
        return out
