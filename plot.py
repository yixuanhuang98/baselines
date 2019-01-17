import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

choices = np.loadtxt('baselines/choices.txt')
N = 50
M = 300
choices_stack = np.empty([N, M])

for i in range(0,N):
    a = choices[0:M]
    print(a.shape)
    choices_stack[i] =a

cmap = colors.ListedColormap(['white', 'blue', 'black'])
bounds = [-1,0.5,1.5,2.5]
norm = colors.BoundaryNorm(bounds, cmap.N)


choices = choices_stack
print(choices.shape)
#print(choices)
im = plt.imshow(choices, cmap=cmap, norm=norm)
plt.axis('off')
#plt.colorbar(im, orientation='horizontal')
plt.show()
