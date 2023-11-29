import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise


class CarRacingTD3Agent(TD3BaseAgent):
    def __init__(self, config):
        super(CarRacingTD3Agent, self).__init__(config)
        # initialize environment
        self.env = CarRacingEnvironment(N_frame=4, test=False)
        self.test_env = CarRacingEnvironment(N_frame=4, test=True)

        # behavior network
        self.actor_net = ActorNetSimple(
            self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
        self.critic_net1 = CriticNetSimple(
            self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
        self.critic_net2 = CriticNetSimple(
            self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
        self.actor_net.to(self.device)
        self.critic_net1.to(self.device)
        self.critic_net2.to(self.device)
        # target network
        self.target_actor_net = ActorNetSimple(
            self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
        self.target_critic_net1 = CriticNetSimple(
            self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
        self.target_critic_net2 = CriticNetSimple(
            self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
        self.target_actor_net.to(self.device)
        self.target_critic_net1.to(self.device)
        self.target_critic_net2.to(self.device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

        # set optimizer
        self.lra = config["lra"]
        self.lrc = config["lrc"]

        self.actor_opt = torch.optim.Adam(
            self.actor_net.parameters(), lr=self.lra)
        self.critic_opt1 = torch.optim.Adam(
            self.critic_net1.parameters(), lr=self.lrc)
        self.critic_opt2 = torch.optim.Adam(
            self.critic_net2.parameters(), lr=self.lrc)

        # choose Gaussian noise or OU noise

        noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
        noise_std = np.full(self.env.action_space.shape[0], 1, np.float32)
        self.noise = OUNoiseGenerator(noise_mean, noise_std)

    def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            state = state.unsqueeze(0)
            noise = self.noise.generate()
            action = self.actor_net(state).cpu().numpy()[0]

            action += noise
            # left right clip
            action[0] = np.clip(action[0], -1, 1)
            # gas
            action[1] = np.clip(action[1], 0, 1)
            # break
            action[2] = np.clip(action[2], 0, brake_rate)

        return action

    def update_behavior_network(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.device)
        ### TODO ###

        q_value1 = self.critic_net1(state, action)
        q_value2 = self.critic_net2(state, action)
        noise = torch.tensor([self.noise.generate() for _ in range(
            self.batch_size)], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.target_actor_net(next_state).cpu().numpy()
            next_action += noise
            next_action = np.clip(next_action, -1, 1)
            next_action = torch.tensor(
                next_action, dtype=torch.float32).to(self.device)

            q_next1 = self.target_critic_net1(next_state, next_action)
            q_next2 = self.target_critic_net2(next_state, next_action)
            q_next = torch.min(q_next1, q_next2)
            q_target = reward + self.gamma * q_next * (1 - done)

        criterion = nn.MSELoss()
        critic_loss1 = criterion(q_value1, q_target)
        critic_loss2 = criterion(q_value2, q_target)

        # optimize critic
        self.critic_net1.zero_grad()
        critic_loss1.backward()
        self.critic_opt1.step()

        self.critic_net2.zero_grad()
        critic_loss2.backward()
        self.critic_opt2.step()

        if self.total_time_step % self.update_freq == 0:
            # update actor
            action = self.actor_net(state)
            q_value = self.critic_net1(state, action)
            actor_loss = -torch.mean(q_value)

            # optimize actor
            self.actor_net.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
