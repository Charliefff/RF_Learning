import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
import gymnasium as gym

from base_agent import OUNoiseGenerator, GaussianNoise
from base_agent import TD3BaseAgent
# from models.CarRacing_model_2 import ActorResNet, CriticResNet
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from car_racing_env import CarRacingEnvironmentLocal, CarRacingEnvironmentRemote



class CarRacingTD3Agent(TD3BaseAgent):
  def __init__(self, config):
    super(CarRacingTD3Agent, self).__init__(config)
    # initialize environment
    self.env_remote = config['env_remote']
    if self.env_remote:
      self.env = CarRacingEnvironmentRemote(server_url=config['server_url'], N_frame=4, test=False)
      self.test_env = CarRacingEnvironmentRemote(server_url=config['server_url'], N_frame=4, test=True)
    else:
      self.env = CarRacingEnvironmentLocal(scenario=config['scenario'], N_frame=4, test=False)
      self.test_env = CarRacingEnvironmentLocal(scenario=config['scenario'], N_frame=4, test=True)
    
    self.observation_space_shape = config['observation_space_shape']
    self.action_space_shape = config['action_space_shape']
    # behavior network init 
    self.actor_net = ActorNetSimple(self.observation_space_shape[0], self.action_space_shape[0], 4)
    # print(self.observation_space_shape[0])
    self.critic_net1 = CriticNetSimple(self.observation_space_shape[0], self.action_space_shape[0], 4)
    self.critic_net2 = CriticNetSimple(self.observation_space_shape[0], self.action_space_shape[0], 4)
    # target network init 
    self.target_actor_net = ActorNetSimple(self.observation_space_shape[0], self.action_space_shape[0], 4)
    self.target_critic_net1 = CriticNetSimple(self.observation_space_shape[0], self.action_space_shape[0], 4)
    self.target_critic_net2 = CriticNetSimple(self.observation_space_shape[0], self.action_space_shape[0], 4)

    self.noise = GaussianNoise(self.action_space_shape[0], 0.0, 1.0)
    
    # transfer behavior network and target network to device
    self.actor_net.to(self.device)
    self.critic_net1.to(self.device)
    self.critic_net2.to(self.device)
    self.target_actor_net.to(self.device)
    self.target_critic_net1.to(self.device)
    self.target_critic_net2.to(self.device)
    
    # load model weight
    self.target_actor_net.load_state_dict(self.actor_net.state_dict())
    self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
    self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
    
    # set optimizer
    self.lra = config["lra"]
    self.lrc = config["lrc"]
    
    self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
    self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
    self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)
    
  
  def decide_agent_actions(self, state, mode='train', sigma=0.0): 
    # based on the behavior (actor) network and exploration noise
    with torch.no_grad():
      if len(state.shape) == 3:
        state = state[None, :, :, :]
      state = torch.tensor(state, dtype=torch.float).cuda()
      action = self.actor_net(state).cpu().data.numpy().squeeze()
      noise = self.noise.generate()
      
      if mode == 'train':
        noise = noise / 2
      
      action = action + sigma * noise

      if mode == 'train':
        action[0] = np.clip(action[0], -1, 1)
        action[0] = self.min_max_normalize(action[0], -1, 1, -0.2, 0.2)


    return action
  
  def min_max_normalize(self, old_value, old_min, old_max, new_min, new_max):
    return ((old_value - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min
    

  def update_behavior_network(self):
    # sample a minibatch of transitions
    state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
    ### TD3 ###


    ## Update Critic ## 
    # Critic Loss: Behavior Net Critic 比 Actor 更常更新
    q_value1 = self.critic_net1(state, action)
    q_value2 = self.critic_net2(state, action)

    # select action a_next from target actor network and add noise for smoothing  
    with torch.no_grad():
      a_next = self.target_actor_net(next_state)
      noise = np.array([self.noise.generate() for _ in range(self.batch_size)])
      noise = torch.tensor(noise, dtype=torch.float).cuda()
      q_next1 = self.target_critic_net1(next_state, a_next + noise)
      q_next2 = self.target_critic_net2(next_state, a_next + noise)
      # select min q value from q_next1 and q_next2 (double Q learning)
      q_target = torch.where(done == 0, reward + self.gamma * torch.min(q_next1, q_next2), reward)
    
    # critic loss function
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

    ## Delayed Actor(Policy) Updates ##
    if self.total_time_step % self.update_freq == 0:
      ## update actor ##

      action = self.actor_net(state)
      q_value1 = self.critic_net1(state, action)
      q_value2 = self.critic_net2(state, action)
      q_value = torch.mean(q_value1 + q_value2)
      actor_loss = -1 * q_value
      # optimize actor
      self.actor_net.zero_grad()
      actor_loss.backward()
      self.actor_opt.step()
    
