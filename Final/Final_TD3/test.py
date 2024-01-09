from environment_wrapper.CarRacingEnv import CarRacingEnvironment, CarRacingEnvironmentRemote
from racecar_gym.env import RaceEnv
import numpy as np
import cv2

from collections import deque

# env = CarRacingEnvironmentRemote(
#     "http://127.0.0.1:1223", N_frame=4, test=False)

env = CarRacingEnvironment(
                N_frame=4, test=False, scenario='circle_cw_competition_collisionStop', render_mode='rgb_array_birds_eye', reset_when_collision=False)

frames = deque(maxlen=4)

obs, info = env.reset()
obs = np.transpose(obs, (1, 2, 0))
obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)  # 96x96
for _ in range(4):
    frames.append(obs)


# info_keys = info.keys()
# for i in info_keys:
#     print(type(info[i]))
obs = np.stack(frames, axis=0)

terminated = False
t = 0
while not terminated:
    action = [1, 0]

    obs, reward, terminated, truncates, info = env.step(action)
    # print(info.keys())
    obs = np.transpose(obs, (1, 2, 0))
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = np.stack(frames, axis=0)
    t += 1
    x, y, z, a, b, c = info['velocity']
    # print('velocity\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(x, y, z))
    print('angular velocity\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(a, b, c))

    # env.render()
print(t)