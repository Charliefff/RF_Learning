import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random


class AtariDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super(AtariDQNAgent, self).__init__(config)
        self.env = gym.make('ALE/MsPacman-v5', render_mode="rgb_array")
        self.test_env = gym.make('ALE/MsPacman-v5', render_mode="rgb_array")
        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(self.env.action_space.n)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(self.env.action_space.n)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(
            self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

    def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            observation = torch.tensor(
                observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.behavior_net(observation).argmax(dim=1).item()

        return action

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.device)

        q_value = self.behavior_net(state)
        q_value = q_value.gather(1, action.to(torch.int64))

        q_next = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # behavior_net 決定動作 q_next_action來給value
            q_next_action = self.behavior_net(next_state).argmax(dim=1)
            q_next = self.target_net(next_state).gather(
                1, q_next_action.to(torch.int64))
            q_target = reward + self.gamma * q_next * (1 - done)

        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
