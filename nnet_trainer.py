import torch
import torch.optim as optim
import numpy as np


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


class NN_trainer:
    def __init__(self, nn):
        self.nn = nn

    def train(self, examples, num_epoch, parameters):
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
                out_pi, out_v = self.nn(states)
                l_pi = loss_pi(target_pis, out_pi)
                l_v = loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # compute gradient and do SGD step
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
