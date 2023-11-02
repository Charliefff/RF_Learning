import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym


class AtariPPOAgent(PPOBaseAgent):
    def __init__(self, config):
        super(AtariPPOAgent, self).__init__(config)
        
        self.env = gym.make(self.env_id, render_mode="rgb_array")
        self.env = gym.wrappers.AtariPreprocessing(
            self.env, 30, 1, 84, terminal_on_life_loss=False, grayscale_obs=True
        )
        self.env = gym.wrappers.FrameStack(self.env, 4)
        ### TODO ###
        # initialize test_env
        self.test_env = gym.make(self.env_id, render_mode="rgb_array")
        self.test_env = gym.wrappers.AtariPreprocessing(
            self.test_env, 30, 1, 84, terminal_on_life_loss=False, grayscale_obs=True
        )
        self.test_env = gym.wrappers.FrameStack(self.test_env, 4)

        self.net = AtariNet(self.env.action_space.n)
        self.net.to(self.device)
        self.lr = config["learning_rate"]
        self.update_count = config["update_ppo_epoch"]
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=lambda epoch: 1 / (1 + epoch * 0.0001))

    def decide_agent_actions(self, observation, eval=False):

        observation = np.asarray(observation)
        observation = torch.tensor(
            observation, dtype=torch.float32, device=self.device)
        observation = observation.unsqueeze(0)

        if eval:
            action, v, logp, _ = self.net(observation, eval=True)

        else:
            action, v, logp, _ = self.net(observation)

        return action, v, logp

    def update(self):
        loss_counter = 0.0001
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(
            self.discount_factor_gamma, self.discount_factor_lambda)
        sample_count = len(batches["action"])
        batch_index = np.random.permutation(sample_count)

        observation_batch = {}
        for key in batches["observation"]:
            observation_batch[key] = batches["observation"][key][batch_index]
        action_batch = batches["action"][batch_index]
        return_batch = batches["return"][batch_index]
        adv_batch = batches["adv"][batch_index]
        v_batch = batches["value"][batch_index]
        logp_pi_batch = batches["logp_pi"][batch_index]

        for _ in range(self.update_count):
            for start in range(0, sample_count, self.batch_size):
                ob_train_batch = {}
                for key in observation_batch:
                    ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
                ac_train_batch = action_batch[start:start + self.batch_size]
                return_train_batch = return_batch[start:start +
                                                  self.batch_size]
                adv_train_batch = adv_batch[start:start + self.batch_size]
                v_train_batch = v_batch[start:start + self.batch_size]
                logp_pi_train_batch = logp_pi_batch[start:start +
                                                    self.batch_size]

                ob_train_batch = torch.from_numpy(
                    ob_train_batch["observation_2d"])
                ob_train_batch = ob_train_batch.to(
                    self.device, dtype=torch.float32)
                ac_train_batch = torch.from_numpy(ac_train_batch)
                ac_train_batch = ac_train_batch.to(
                    self.device, dtype=torch.long)
                adv_train_batch = torch.from_numpy(adv_train_batch)
                adv_train_batch = adv_train_batch.to(
                    self.device, dtype=torch.float32)
                v_train_batch = torch.from_numpy(v_train_batch)
                v_train_batch = v_train_batch.to(
                    self.device, dtype=torch.float32)

                logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
                logp_pi_train_batch = logp_pi_train_batch.to(
                    self.device, dtype=torch.float32)
                return_train_batch = torch.from_numpy(return_train_batch)
                return_train_batch = return_train_batch.to(
                    self.device, dtype=torch.float32)

                _, v, logp_pi, entropy = self.net(ob_train_batch)

                ratio = torch.exp(logp_pi - logp_pi_train_batch)
                surr_loss = ratio * adv_train_batch
                surr_loss2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                         1.0 + self.clip_epsilon) * adv_train_batch
                surrogate_loss = -torch.mean(torch.min(surr_loss, surr_loss2))
                value_criterion = nn.MSELoss()

                v_loss_unclip = value_criterion(v, return_train_batch)
                v_clipped = v_train_batch + \
                    torch.clamp(v - v_train_batch, -
                                self.clip_epsilon, self.clip_epsilon)
                v_loss_clipped = value_criterion(v_clipped, return_train_batch)
                v_loss = 0.5 * torch.max(v_loss_unclip, v_loss_clipped)

                loss = surrogate_loss + self.value_coefficient * \
                    v_loss - self.entropy_coefficient * entropy.mean()

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_gradient_norm)
                self.optim.step()
                self.scheduler.step()

                total_surrogate_loss += surrogate_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                loss_counter += 1

        self.writer.add_scalar('PPO/Loss', total_loss /
                               loss_counter, self.total_time_step)
        self.writer.add_scalar(
            'PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
        self.writer.add_scalar(
            'PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
        self.writer.add_scalar(
            'PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
        print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
