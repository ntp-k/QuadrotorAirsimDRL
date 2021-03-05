import numpy as np
import math
import time

import gym
from gym import spaces
from airsim.AirSimClient import *
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.drone_env import AirSimDroneEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# connect to the AirSim simulator
# client = MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)
# client.armDisarm(True)

env = AirSimDroneEnv()


