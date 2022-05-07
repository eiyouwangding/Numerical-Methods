import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from scipy.stats import linregress

font = {
         'size'   : 12
         }

matplotlib.rc('font', **font)

plate_length = 1
T = 0.1
left = 0
right = 0
k = 1

delta_x = 0.025
delta_t = 0.025

x = np.arange(0, plate_length+delta_x, delta_x).round(3)
t = np.arange(0, T+delta_t, delta_t).round(3)
N = len(x)
M = len(t)
T = np.zeros((N, M))

# set initial condition and boundary condition
T[0,:] = left
T[-1,:] = right
T[:, 0] = 6*np.sin(np.pi*x)

gamma = k*delta_t/(delta_x**2)
for j in range(1, M):
    for i in range(1, N-1):
        T[i, j] = gamma*T[i-1, j-1] +(1-2*gamma)*T[i, j-1]+gamma*T[i+1, j-1]

R = np.linspace(1, 0, M)
B = np.linspace(0, 1, M)
G = 0
for j in range(M):
    plt.plot(x, T[:, j], color = [R[j], G, B[j]])
    
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.legend([f't = {value}' for value in t])
plt.xlabel(r'$x$', size=13)
plt.ylabel(r'$u(x, t)$', size=13)
plt.title(r'Plot of $u(x, t)$ $(\Delta x =0.025, \Delta t = 0.25)$')
plt.savefig('EulerOned.jpeg', dpi=1000, bbox_inches='tight')