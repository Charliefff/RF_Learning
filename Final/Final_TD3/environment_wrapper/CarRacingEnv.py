import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from racecar_gym.env import RaceEnv
import numpy as np
import requests
import json


class CarRacingEnvironmentRemote:
    def __init__(self, server_url, N_frame=4, test=False):
        self.server_url = server_url
        self.frames = deque(maxlen=N_frame)
        self.test = test
        self.ep_len = 0

    def reset(self):
        response = requests.get(f'{self.server_url}')
        if json.loads(response.text).get('error'):
            raise requests.exceptions.RequestException(
                json.loads(response.text)['error'])
        else:
            result = json.loads(response.text)
            obs = result['observation']
            obs = np.array(obs).astype(np.uint8)
            obs = obs.transpose(1, 2, 0)
            obs = cv2.resize(obs, (128, 128))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            for _ in range(self.frames.maxlen):
                self.frames.append(obs)

            obs = np.stack(self.frames, axis=0)
            # API沒有提供info
            return obs, None

    def step(self, action):
        # 給定 action, 然後回傳 state
        if (type(action) == list) or (type(action) == np.ndarray or type(action) == tuple):
            response = requests.post(f'{self.server_url}', json={
                'action': action})
        else:
            response = requests.post(f'{self.server_url}', json={
                'action': action.tolist()})
        if json.loads(response.text).get('error'):
            raise requests.exceptions.RequestException(
                json.loads(response.text)['error'])
        else:
            result = json.loads(response.text)
            # print(result)
            # obs = result['observation']
            # reward = result['reward']
            terminal = result['terminal']
            if terminal:
                obs = None
            else:
                # trunc = result['trunc']
                response = requests.get(f'{self.server_url}')

                if json.loads(response.text).get('error'):
                    raise requests.exceptions.RequestException(
                        json.loads(response.text)['error'])
                else:
                    result = json.loads(response.text)
                    obs = result['observation']
                    obs = np.array(obs).astype(np.uint8)
                    obs = obs.transpose(1, 2, 0)
                    obs = cv2.resize(obs, (128, 128))
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                    self.frames.append(obs)
                    obs = np.stack(self.frames, axis=0)

            return obs, None, terminal, None, None


class CarRacingEnvironment:
    def __init__(self, N_frame=4, test=False, scenario='circle_cw', render_mode='rgb_array_birds_eye', reset_when_collision=True):
        self.test = test
        if self.test:
            self.env = RaceEnv(scenario=scenario, render_mode=render_mode,
                               reset_when_collision=reset_when_collision)
        else:
            self.env = RaceEnv(scenario=scenario, render_mode=render_mode,
                               reset_when_collision=reset_when_collision)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.ep_len = 0
        self.frames = deque(maxlen=N_frame)
        self.frames1 = None
        self.frames2 = None
        self.goal = None

    def step(self, action):
        obs, reward, terminates, truncates, info = self.env.step(action)
        original_reward = reward
        original_terminates = terminates
        self.ep_len += 1

        obs = np.transpose(obs, (1, 2, 0))
        # convert to grayscale
        obs = cv2.resize(obs, (128, 128))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        # save image for debugging
        # filename = "/data/tzeshinchen/RF_Learning/Final/environment_wrapper/images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, obs)

# ********************************************************************************

        # frame stacking
        if self.ep_len == 1:
            self.frames1 = self.frames
            self.frames2 = self.frames

        if self.ep_len % 2:
            self.frames1.append(obs)
            self.frames = self.frames1
        else:
            self.frames2.append(obs)
            self.frames = self.frames2

        obs = np.stack(self.frames, axis=0)

        if self.test:
            reward = original_reward

        # foward

        reward *= 100
        # print(reward)
        if self.goal is None:
            self.goal = info['dist_goal']

        else:
            if self.goal - info['dist_goal'] > 0:
                if info['obstacle'] < 0.1:
                    reward -= 0.1
            else:
                reward -= 0.01

        self.goal = info['dist_goal']
        
        if info['obstacle'] > 0.8:
            reward += 0.1
        

# ********************************************************************************

        return obs, reward, terminates, truncates, info

    def reset(self):
        obs, info = self.env.reset()
        self.ep_len = 0
        obs = np.transpose(obs, (1, 2, 0))
        obs = cv2.resize(obs, (128, 128))

        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

# ************************************************************

        # frame stacking
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)

# ************************************************************

        return obs, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == '__main__':

    env = CarRacingEnvironment(test=True)
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_length = 0
    t = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        obs, reward, terminates, truncates, info = env.step(action)
        print(
            f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
        total_reward += reward
        total_length += 1
        env.render()
        if terminates or truncates:
            done = True

    print("Total reward: ", total_reward)
    print("Total length: ", total_length)
    env.close()
