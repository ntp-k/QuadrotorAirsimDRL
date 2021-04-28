
import gym
from gym import spaces

from airsim.AirSimClient import *
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.drone_env import AirSimDroneEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.env_checker import check_env

import numpy as np


# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
destination = np.array([70,-5,-20])

env = gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=1,
                image_shape=(84, 84, 1),
                destination=destination,
            )

check_env(env)


# env = DummyVecEnv(
#     [
#         lambda: Monitor(
#             gym.make(
#                 "airgym:airsim-drone-sample-v0",
#                 ip_address="127.0.0.1",
#                 step_length=1,
#                 image_shape=(84, 84, 1),
#                 destination=destination,
#             )
#         )
#     ]
# )

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)
