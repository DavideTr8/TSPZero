import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_len, output_len):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_len, 64)
        self.fc2 = nn.Linear(64, output_len - 1)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        p = F.softmax(self.fc2(x), dim=1)
        v = F.tanh(self.fc3(x))
        return p, v
