import numpy as np
import matplotlib.pyplot as plt

rewards = np.loadtxt('./data/scn_rew.txt')
ts = np.loadtxt('./data/scn_ts.txt')
mlp_rewards = np.loadtxt('./data/mlp64_rew.txt')
mlp_ts = np.loadtxt('./data/mpl64_ts.txt')
rewards = rewards * 10
mlp_rewards = mlp_rewards * 10
plt.plot(ts,rewards)
plt.plot(mlp_ts,mlp_rewards)
plt.show()
