import numpy as np
import matplotlib.pyplot as plt

rewards = np.loadtxt('./data/scn_rew.txt')
ts = np.loadtxt('./data/scn_ts.txt')
rewards = rewards * 10
plt.plot(ts,rewards)
plt.show()
