import numpy as np
import matplotlib.pyplot as plt

rewards = np.loadtxt('./data/fcn_swim_rew.txt')
ts = np.loadtxt('./data/fcn_swim_ts.txt')
mlp_rewards = np.loadtxt('./data/mlp_swim_rew.txt')
mlp_ts = np.loadtxt('./data/mlp_swim_ts.txt')
rewards = rewards * 10
mlp_rewards = mlp_rewards * 10
plt.plot(ts,rewards)
plt.plot(mlp_ts,mlp_rewards)
plt.show()
