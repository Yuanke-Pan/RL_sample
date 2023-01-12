import gym
from stable_baselines3 import DQN

env = gym.make("Pong-v4")

model = DQN("CnnPolicy", env, verbose=1,buffer_size=10000)
model.learn(total_timesteps=1000000, log_interval=4)
model.save('dqn_pong')

del model # remove to demonstrate saving and loading