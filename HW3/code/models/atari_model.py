import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                 nn.ReLU(True),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(True),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                 )
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                           nn.ReLU(True),
                                           nn.Linear(512, num_classes)
                                           )
        
        # 有更改value 網路
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                   nn.Tanh(),
                                   nn.Linear(512, 256),
                                   nn.Tanh(),
                                   nn.Linear(256, 1)
                                   )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, eval=False):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        # actor 網路
        logits = self.action_logits(x)

        # value 網路
        value = self.value(x)
        value = torch.squeeze(value)

        dist = Categorical(logits=logits) #得到機率分布

        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        return NotImplementedError