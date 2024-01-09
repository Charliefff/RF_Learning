from collections import deque

import requests
import json
import gymnasium as gym
import numpy as np
import cv2
import time
from datetime import datetime, timedelta
from racecar_gym.env import RaceEnv


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
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            for _ in range(self.frames.maxlen):
                self.frames.append(obs)

            obs = np.stack(self.frames, axis=0)
            # API沒有提供info
            return obs, None

    def step(self, action):
        # 給定 action, 然後回傳 state
        response = requests.post(f'{self.server_url}', json={
                                 'action': action.tolist()})
        if json.loads(response.text).get('error'):
            raise requests.exceptions.RequestException(
                json.loads(response.text)['error'])
        else:
            result = json.loads(response.text)
            terminal = result['terminal']
            if terminal:
                obs = None

            else:

                response = requests.get(f'{self.server_url}')

                if json.loads(response.text).get('error'):
                    raise requests.exceptions.RequestException(
                        json.loads(response.text)['error'])
                else:
                    result = json.loads(response.text)
                    obs = result['observation']
                    obs = np.array(obs).astype(np.uint8)
                    obs = obs.transpose(1, 2, 0)
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                    self.frames.append(obs)
                    obs = np.stack(self.frames, axis=0)

            return obs, None, terminal, None, None


class CarRacingEnvironmentLocal:
    def __init__(self, scenario, N_frame=4, test=False):
        self.test = test

        if scenario == 'austria_competition':
            self.reset_when_collision = True
        else:
            self.reset_when_collision = False

        if self.test:
            self.env = RaceEnv(
                scenario=scenario,
                render_mode='rgb_array_birds_eye',
                reset_when_collision=self.reset_when_collision,
            )
        else:
            self.env = RaceEnv(
                scenario=scenario,
                reset_when_collision=self.reset_when_collision,
            )

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.ep_len = 0
        self.frames = deque(maxlen=N_frame)
        self.frames1 = deque(maxlen=N_frame)
        self.frames2 = deque(maxlen=N_frame)
        self.last_progress = 0
        self.last_checkpoint = 0
        self.max_delta = 0.0011
        self.max_speed = 0

    def step(self, action):

        obs, reward, terminates, truncates, info = self.env.step(action)
        self.ep_len += 1

        reward *= 100
        foward, _, _, _, _, _ = info['velocity']

        # if foward >= self.max_speed:
        #     self.max_speed = foward
        #     reward += 0.05
        # elif foward > 0 and foward < self.max_speed:
        #     reward = (foward / self.max_speed) * 0.04
        # else:
        #     reward = 0

        delta = abs(info['progress'] - self.last_progress)
        if delta > 0:
            # progress_reward = (delta / self.max_delta) * 1.2
            progress_reward = 0.6
        else:
            progress_reward = -0.1

        if info['checkpoint'] != self.last_checkpoint:
            checkpoint_reward = 100
        else:
            checkpoint_reward = 0

        if info['obstacle'] > 0.8:
            obstacle_penalty += 1.0
        elif info['obstacle'] < 0.2:
            obstacle_penalty = -0.5
        else:
            obstacle_penalty = 0

        # if delta > self.max_delta:
        #     self.max_delta = delta

        reward = reward + progress_reward + obstacle_penalty + checkpoint_reward

        self.last_progress = info['progress']
        self.last_checkpoint = info['checkpoint']

        obs = obs.transpose(1, 2, 0)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # # 2 frame stack
        # if self.ep_len == 1:
        #     self.frames1 = self.frames
        #     self.frames2 = self.frames

        # if self.ep_len % 2 == 0:
        #     self.frames1.append(obs)
        #     self.frames = self.frames1
        # else:
        #     self.frames2.append(obs)
        #     self.frames = self.frames2

        self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)

        return obs, reward, terminates, truncates, info

    def reset(self):
        self.acc_time = timedelta(seconds=0)
        self.ep_len = 0
        obs, info = self.env.reset()

        self.last_progress = info['progress']
        obs = obs.transpose(1, 2, 0)  # (128,128,3)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)

        obs = np.stack(self.frames, axis=0)
        return obs, info

    def close(self):
        self.env.close()


if __name__ == '__main__':
    # Remote Env Test
    # server_url = 'http://127.0.0.1:1224'
    # action_space = gym.spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.float32)
    # rand_agent = CarRacingEnvironmentRemote(
    #   server_url = server_url,
    #   action_space = action_space
    # )
    # done = False
    # total_reward = 0
    # total_length = 0
    # t = 0

    # while not done:
    #   t += 1
    #   print("Step: ", t)
    #   action = rand_agent.action_space.sample()
    #   try:
    #     # state, reward, terminal, trunc = rand_agent.step(action)
    #     state = rand_agent.reset()
    #   except requests.exceptions.RequestException as e:
    #     print('HTTP ERROR')
    #     print('ERROR MESSAGE: ', e)
    #   total_reward += reward
    #   total_length += 1
    #   if terminal or trunc:
    #     done = True

    # print("Total reward: ", total_reward)
    # print("Total length: ", total_length)

    # Local Env Test
    env = CarRacingEnvironmentLocal(scenario='austria_competition', test=True)
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_length = 0
    t = 0
    while not done:
        t += 1
        # motor（馬達）and steering（轉向）, 兩個都介於(-1,1), 馬達應該包含煞車
        action = env.action_space.sample()
        obs, reward, terminates, truncates, info = env.step(action)
        print(info)
        total_reward += reward
        total_length += 1
        if terminates or truncates:
            done = True

    print("Total reward: ", total_reward)
    print("Total length: ", total_length)
    env.close()
