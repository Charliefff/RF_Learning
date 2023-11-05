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
        
        # actor
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, num_classes)
                                        )
        
        # critic
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 1)
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, new_action=None, noise_std=0.1):
        
        x = x.float() / 255.
        if not eval:
            noise = torch.randn_like(x) * noise_std
            x = x + noise 
        
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        
        value = self.value(x)
        value = torch.squeeze(value)

        logits = self.action_logits(x)
        
        dist = Categorical(logits=logits)

        if eval:
            action = torch.argmax(logits, dim=1)
        else:
            if new_action is None:
                action = dist.sample()
            else:
                action = new_action
        
        return action, value, dist.log_prob(action), dist.entropy()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                