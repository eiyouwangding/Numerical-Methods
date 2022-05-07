import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from scipy.stats import linregress

font = {
         'size'   : 12
         }

matplotlib.rc('font', **font)

delta_t = 0.01
x_list = [1]
max_iter_times = int(10/delta_t)
t_list = np.linspace(0, 10, max_iter_times+1)
x= np.e**(-2*t_list)
for i in range(max_iter_times):
    x0 = x_list[-1]
    x1 = x0+(-2*x0)*delta_t
    x_list.append(x1)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.plot(t_list, x, label='Analytical Solution')
#plt.scatter(t_list, x_list, c='red', marker='o', s=1, label='Numerical Soltuion')
plt.plot(t_list, x_list, label='Numerical Solution')
#plt.legend(loc='center left', bbox_to_anchor=(0.25, 1.25))
plt.legend()
plt.xlabel(r'$t$', size=13)
plt.ylabel(r'$x$', size=13)
plt.title(r'Plot of $x$ $(\Delta t=%s)$'%str(delta_t))
plt.savefig('Euler_%s.jpeg'%str(delta_t), dpi=1000, bbox_inches='tight')