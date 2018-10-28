import torch
import torch.nn as nn
import torch.nn.functional as F

class IndRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(IndRNNCell, self).__init__()
        self.input_fwd = nn.Linear(input_size, hidden_size)
        self.hidden_vector = nn.Parameter(torch.rand(1, hidden_size) - 0.5)
        self.bias = nn.Parameter(torch.rand(hidden_size) - 0.5)
        self.activation = F.relu

    def forward(self, x, h):
        return self.activation(self.input_fwd(x) + self.hidden_vector * h + self.bias)


class SingleLayerIndRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleLayerIndRNN, self).__init__()
        self.cell = IndRNNCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        input_count = x.size(0)
        h = torch.zeros(input_count, self.input_size)

        for i in range(x.size(1)):
            x_i = x[:, i, :]
            h = self.cell(x_i, h)

        return h
