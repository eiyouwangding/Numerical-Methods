import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math
import scipy
from scipy.stats import linregress
import scipy.integrate as integrate

font = {
         'size'   : 12
         }

matplotlib.rc('font', **font)

plate_length = 1
max_iter_time = 1700

k = 1
delta_x = 0.02
grid_number = int(plate_length/delta_x)

delta_t = (delta_x ** 2)/(4 * k)
gamma = (k * delta_t) / (delta_x ** 2)

u = np.empty((max_iter_time, grid_number, grid_number))

# initial condition
u_initial = 0

# boundary condition
u_top = 1.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# set initial and boundary condition
u.fill(u_initial)

u[:, (grid_number-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (grid_number-1):] = u_right


def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, grid_number-1):
            for j in range(1, grid_number-1):
                u[k + 1, j, i] = gamma * (u[k][j+1][i] + u[k][j-1][i] + u[k][j][i+1] + u[k][j][i-1] - 4*u[k][j][i]) + u[k][j][i]
        if k%200 == 0:
            plt.pcolormesh(u[k], cmap=plt.cm.jet, vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel('x', size=13)
            plt.ylabel('y', size=13)
            plt.xticks(size = 12)
            plt.yticks(size = 12)
            time = str(k*delta_t)
            plt.savefig('Euler2D/2d_%s.jpeg'%time, dpi=1000, bbox_inches='tight')
            plt.show()

    return u
u = calculate(u)