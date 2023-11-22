import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
                                 nn.ReLU(True),
                                 layer_init(
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                                 nn.ReLU(True),
                                 layer_init(
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                                 nn.ReLU(True),
                                 nn.Flatten(),
                                 layer_init(nn.Linear(7*7*64, 512)),
                                 nn.ReLU(True)
                                 )

        # actor
        self.action_logits = nn.Sequential(
            layer_init(nn.Linear(512, num_classes), std=0.01)
        )

        # critic
        self.value = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1.0)
        )

    def forward(self, x, eval=False, new_action=None):

        x = x.float() / 255.
        x = self.cnn(x)
        value = self.value(x)
        value = torch.squeeze(value)

        logits = self.action_logits(x)

        dist = Categorical(logits=logits)

        if eval:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()

        return action, value, dist.log_prob(action), dist.entropy()
