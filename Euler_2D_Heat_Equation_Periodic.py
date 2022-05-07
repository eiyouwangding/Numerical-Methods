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

def jinteger(i):
    if str(i).split('.')[-1] =='0':
        return True
    else:
        return False
    
def plotheatmap(u_k, k, vmin, vmax):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=1)
    plt.colorbar()

    return plt

class PeriodicSolver:
    
    def __init__(self, plate_length, max_iter_time, delta_x, k, top, bottom, ic_func):
        self.plate_length = plate_length
        self.max_iter_time = max_iter_time
        self.delta_x = delta_x
        self.k = k
        self.ic_func = ic_func
        
        self.delta_t = (self.delta_x )/(4 * self.k * 100)
        self.gamma = (self.k * self.delta_t) / (self.delta_x ** 2)
        
        self.top = top
        self.bottom = bottom
    
    def initialize(self):
        if jinteger(self.plate_length/self.delta_x):
            self.grid_number = int(self.plate_length/self.delta_x)+1
        else:
            print('pick another delta x, initialization fails')
            return False
        self.u = np.empty((self.max_iter_time, self.grid_number, self.grid_number))
        return self.u
        
    def set_initial_conditions(self):
        temp = np.linspace(0, self.plate_length, self.grid_number)
        column = eval(self.ic_func)
        for i in range(self.grid_number):
            self.u[:,:,i] = column
        return self.u
        
    def set_boundary_conditions(self):
        self.u[:, self.grid_number-1:, :] = self.top
        self.u[:, 0, :] = self.bottom
        return self.u
    
    def calculate_periodic(self):
        for k in range(0, self.max_iter_time-1):
            for i in range(0, self.grid_number-1):
                if i==0:
                    for j in range(1, self.grid_number-1):
                        self.u[k+1, j, i] = self.gamma * (self.u[k][j+1][i] + self.u[k][j-1][i] + self.u[k][j][i+1] + self.u[k][j][self.grid_number-2] - 4*self.u[k][j][i]) + self.u[k][j][i]
                        self.u[k+1, j, self.grid_number-1] =  self.u[k+1, j, i]
                else:
                    for j in range(1, self.grid_number-1):
                        self.u[k + 1, j, i] = self.gamma * (self.u[k][j+1][i] + self.u[k][j-1][i] + self.u[k][j][i+1] + self.u[k][j][i-1] - 4*self.u[k][j][i]) + self.u[k][j][i]
            
            if k%500 == 0:
                plt.pcolormesh(pde.u[k], cmap=plt.cm.jet, vmin=0, vmax=1)
                plt.colorbar()
                time = str(k*self.delta_t)
                plt.xlabel(r'$x$', size=13)
                plt.ylabel(r'$y$', size=13)
                plt.xticks(size = 12)
                plt.yticks(size = 12)
                #plt.title(r'$\Delta t = %s$'%time)
                plt.savefig('periodic2/periodic2_%s.jpeg'%time, dpi=1000, bbox_inches='tight')
                plt.show()

        #return self.u
    
    def animation(self, file_name):
        def _animate(k):
            plotheatmap(self.u[k], k, vmin=0, vmax=1)

        anim = animation.FuncAnimation(plt.figure(), _animate, interval=1, frames=self.max_iter_time, repeat=False)
        anim.save(file_name)
        
    def compute_sol(self, solutions):
        u_true = np.empty((self.max_iter_time, self.grid_number, self.grid_number))
        for k in range(self.max_iter_time):
            t = k*self.delta_t
            y = np.linspace(0, self.plate_length, self.grid_number)
            y = eval(solutions)
            for i in range(self.grid_number):
                u_true[k,:,i] = y
        return u_true
        
    def main(self, animation):
        self.initialize()
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.calculate_periodic() 
        
        if animation != False:
            self.animation(animation)
        
        #return self.u
        
if __name__=='__main__':
    pde = PeriodicSolver(plate_length=1, max_iter_time=2600, delta_x=0.01, k=1, top=0, bottom=0, ic_func='np.sin((temp*math.pi)/self.plate_length)')
    pde.main(animation=False)