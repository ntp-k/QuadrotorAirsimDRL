from stable_baselines3 import A2C

model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=100, tb_log_name="second_run")
# Pass reset_num_timesteps=False to continue the training curve in tensorboard
# By default, it will create a new curve
model.learn(total_timesteps=10000, tb_log_name="second_run", reset_num_timesteps=False)
model.learn(total_timesteps=10000, tb_log_name="third_run", reset_num_timesteps=False)


"""
enter in ternimal



tensorboard --logdir=./tb_logs/ --host=127.0.0.1  
tensorboard --logdir='/Users/natthaphat/Work/CPE/Senior/QuadrotorAirsimDRL/a2c_cartpole_tensorboard' --host=127.0.0.1  

tensorboard --inspect --logdir /Users/natthaphat/Work/CPE/Senior/QuadrotorAirsimDRL/tb_logs
tensorboard --inspect --logdir /Users/natthaphat/Work/CPE/Senior/QuadrotorAirsimDRL/a2c_cartpole_tensorboard


"""