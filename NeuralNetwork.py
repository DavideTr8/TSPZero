import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class NN(nn.Module):
    def __init__(self, mlp_net, conv_net):
        super(NN, self).__init__()
        self.mlp_net = mlp_net
        self.conv_net = conv_net

    def forward(self, distances, one_hot):
        feature_matrix = self.conv_net(distances)
        feature_prime = feature_matrix.flatten()
        C = torch.cat((feature_prime, one_hot), dim=1)
        p, v = self.mlp_net(C)
        return p, v


class MLP_NN(nn.Module):
    def __init__(self, input_dim, policy_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, policy_dim)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        p = F.softmax(self.fc2(x), dim=0)
        v = torch.tanh(self.fc3(x))
        return p, v


class Conv_NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
        self.max_p = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.max_p(self.conv1(x))
        return x


def train(neuralnet, examples, num_epoch, parameters):
    """
    examples: list of examples, each example is of form (state, pi, v)
    """
    batch_size = 10
    optimizer = optim.Adam(parameters())

    for epoch in range(num_epoch):
        batch_count = int(len(examples) / batch_size)

        for _ in range(batch_count):
            sample_ids = np.random.randint(len(examples), size=batch_size)
            states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            states = torch.FloatTensor(np.array(states).astype(np.float64))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

            # compute output
            out_pi, out_v = neuralnet(states)
            l_pi = loss_pi(target_pis, out_pi)
            l_v = loss_v(target_vs, out_v)
            total_loss = l_pi + l_v

            # compute gradient and do SGD step
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]