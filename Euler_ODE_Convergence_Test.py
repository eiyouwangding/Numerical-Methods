import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from scipy.stats import linregress

delta_t_list = [0.001, 0.005, 0.01, 0.02]
error_list = []
for delta_t in delta_t_list:
    x_list = [1]
    max_iter_times = int(10/delta_t)
    t_list = np.linspace(0, 10, max_iter_times+1)
    x= np.e**(-2*t_list)
    for i in range(max_iter_times):
        x0 = x_list[-1]
        x1 = x0+(-2*x0)*delta_t
        x_list.append(x1)
    error = np.linalg.norm(x-x_list, ord=np.inf)
    error_list.append(error)

plt.xticks(size = 12)
plt.yticks(size = 12)
plt.plot(np.log(delta_t_list), np.log(error_list))
plt.scatter(np.log(delta_t_list), np.log(error_list))
plt.xlabel(r'$\log (\Delta t)$', size=13)
plt.ylabel(r'$\log (error)$', size=13)
plt.title(r'$\log (error)-\log (\Delta t)$')
linregress(np.log(delta_t_list), np.log(error_list))

plt.savefig('Euler_test.jpeg', dpi=1000, bbox_inches='tight')