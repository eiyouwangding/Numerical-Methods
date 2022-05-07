import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math
import scipy
from scipy.stats import linregress
import scipy.integrate as integrate

# numerical solver for 1D heat equation with homogeneous Dirichlet boundary condition
class OneDBackEulerSolver:
    
    def __init__(self, plate_length, k, delta_x, delta_t, max_iter_time, left, right, u0):
        self.plate_length = plate_length
        self.k = k
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.max_iter_time = max_iter_time
        self.left = left
        self.right = right
        self.u0 = u0
        
        self.gamma = self.k*self.delta_t/(self.delta_x**2)
        self.grid_number = int(self.plate_length/self.delta_x)+1
        
        self.x = np.linspace(0, self.plate_length, self.grid_number)
    
    def initialize(self):
        # initialize matrix A
        self.A = np.zeros((self.grid_number-2, self.grid_number-2))
        for i in range(self.A.shape[0]):
            if i == 0:
                self.A[i, i] = 1+2*self.gamma
                self.A[i, i+1] = -self.gamma
            elif i == self.A.shape[0]-1:
                self.A[i, i] = 1+2*self.gamma
                self.A[i, i-1] = -self.gamma
            else:
                self.A[i, i] = 1+2*self.gamma
                self.A[i, i-1] = -self.gamma
                self.A[i, i+1] = -self.gamma
        
        self.A_inv = np.linalg.inv(self.A)
        
        # initialize matrix u
        self.u = np.zeros((self.grid_number, self.max_iter_time))
        #self.u[0, :] = self.left
        #self.u[-1, :] = self.right
        #self.u[1:-1, 0] = self.u0
        
        self.u[:, 0] = 6*np.sin(np.pi*self.x)
        self.u[0, :] = self.left
        self.u[-1, :] = self.right
        
    def calculate(self):
        # solve Ax=b
        for k in range(1, self.max_iter_time):
            temp = self.u[1:-1, k-1]
            temp[0] = temp[0] +self.gamma*self.left
            temp[-1] = temp[-1] +self.gamma*self.right
            self.u[1:-1, k] = np.matmul(self.A_inv, temp)
            #self.u[1:-1, k] = scipy.linalg.solve(self.A, temp)
            
    def analy_solution(self):
        '''u_analy = np.zeros((self.grid_number, self.max_iter_time))
        for k in range(self.max_iter_time):
            t = k*self.delta_t
            for i in range(1, 101, 2):
                # compute bn
                #integral = integrate.quad(lambda x: self.u0*np.sin(i*math.pi*x/self.plate_length), 0, self.plate_length)
                #bn = 2/self.plate_length*integral[0]
                #u_analy[:, k] += bn*np.sin(i*math.pi*self.x)*math.e**(-(i*math.pi)**2*t)
                u_analy[:, k] += (2/(i*math.pi))*np.sin(i*math.pi*self.x)*math.e**(-(i*math.pi)**2*t)'''
        u_analy = np.zeros((self.grid_number, self.max_iter_time))
        for k in range(self.max_iter_time):
            t = k*self.delta_t
            u_analy[:, k] = 6*np.sin(np.pi*self.x)*(np.e**(-self.k*t*(np.pi/self.plate_length)**2))
        return u_analy
    
    def main(self):
        self.initialize()
        self.calculate()
        
if __name__=='__main__':
    # convergence test
    delta_x_list = [0.0005, 0.001, 0.002, 0.005]
    t = 50
    error_list = []
    for j in range(len(delta_x_list)):
        delta_x = delta_x_list[j]
        pde = OneDBackEulerSolver(plate_length=1, k=1, delta_x=delta_x, delta_t=0.00000001, max_iter_time=100, left=0, right=0, u0=0.5)
        pde.main()
        u = pde.u
        u_analy = pde.analy_solution()
        e = np.linalg.norm(u_analy[:, t]-u[:, t], ord=np.inf)
        error_list.append(e)
    plt.plot(np.log(delta_x_list), np.log(error_list))
    linregress(np.log(delta_x_list), np.log(error_list))