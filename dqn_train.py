import gym
from gym import spaces

from airsim.AirSimClient import *
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.drone_env import AirSimDroneEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, EveryNTimesteps

# from stable_baselines3 import results_plotter
# from stable_baselines3.results_plotter import load_results, ts2xy

from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt
from tensorboard import program
import numpy as np
import math
import time
import os

# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#configuration
destination = np.array([70,-5,-20])
time_steps = 5
log_path = './tb_logs/'

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=1,
                image_shape=(84, 84, 1),
                destination=destination,
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# env = Monitor(env, log_path)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    tensorboard_log=log_path,
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.cum_rew = 0

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/cum_rew", self.cum_rew)

        # reset vars once recorded
        self.cum_rew = 0
    
    def _on_step(self) -> bool:
        # log reward
        self.cum_rew += (self.training_env.get_attr("rewards")[0])[-1]
        return True

reward_callback = EveryNTimesteps(
    n_steps=1,
    callback = TensorboardCallback()
)

# callbacks.append(eval_callback)
# callbacks.append(reward_callback)



kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=int(time_steps),
    tb_log_name="dqn_" + str(time_steps) + "_time_steps",
    reset_num_timesteps=False,
    **kwargs
)

model.learn(
    total_timesteps=int(time_steps),
    tb_log_name="test_" + str(time_steps) + "_time_steps",
    callback=TensorboardCallback()
)



# results_plotter.plot_results([log_path], time_steps, results_plotter.X_TIMESTEPS, "DQN_train")
# plt.show()

# Save policy weights
model.save("model/dqn_airsim_drone_policy")