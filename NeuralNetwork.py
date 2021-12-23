import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_len):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_len, 64)
        self.fc2 = nn.Linear(64, input_len)
        self.fc3 = nn.Linear(64, 1)


    def forward(self, input):
        x = F.relu(self.fc1(input))
        p = F.softmax(self.fc2(x), dim=0)
        v = torch.tanh(self.fc3(x))
        return p, v

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        batch_size = 10
        optimizer = optim.Adam(self.parameters())

        for epoch in range(100):
            batch_count = int(len(examples) / batch_size)

            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # compute output
                out_pi, out_v = self(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v


                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]