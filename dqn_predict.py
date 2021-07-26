import gym
from gym import spaces

import airsim
from airgym.envs.airsim_env import AirSimEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=1,
                image_shape=(84, 84, 1),
                destination=np.array([70,-5,-20]),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# model = DQN.load("model/dqn_airsim_drone_policy")
model = DQN.load("checkpoint/v18_dqn_cnnPolicy_4actions_imageObs_100000_steps/dqn_policy_65000_steps.zip")



mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)

print(f"mean_reward = [{mean_reward:.2f}] +/- {std_reward}")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#       obs = env.reset()
#       break



'''
mean_reward, std_reward = evaluate_policy(default_model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
'''