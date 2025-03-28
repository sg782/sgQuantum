import math
import cmath
import numpy as np
import matplotlib.pyplot as plt


def gaussian_wave_func_2d(x,y,t,a=1,m=1,h_bar=1,kx_0=1, ky_0=1):

    val_x = gaussian_wave_func_1d(x,t,a,m,h_bar,kx_0)
    val_y = gaussian_wave_func_1d(y,t,a,m,h_bar,ky_0)

    out = val_x * val_y
    return out

def gaussian_wave_func_1d(x,t,a=1,m=1,h_bar=1,k_0=1):
    # a = initial width
    # m = mass
    # h_bar = reduced plank constant 
    # k_0 = wave number (momentum)
    #

    theta = math.atan((2*h_bar*t)/(m*(a**2)))/2

    phi = - theta - (t * (h_bar * (k_0 **2))/(2*m)) # constant


    p1 = (2 * (a**2)/math.pi) ** (1/4) # constant

    p2_num = np.exp(1j*phi) * np.exp(1j*k_0*x)
    p2_den = ((a**4) + (4*(h_bar**2) * (t**2)) / (m**2))**(1/4)

    p2 = p2_num / p2_den

    p3_num = - (x - (h_bar* k_0*t)/m) **2
    p3_den = (a**2) + (2*1j * h_bar * t) / m

    p3 = np.exp(p3_num/p3_den)

    return p1 * p2 * p3

def plot_wave_func_2d(values):

    prob = abs(values)**2
    
    plt.figure(figsize=(6, 5))
    plt.imshow(prob, origin='lower', cmap='viridis')
    plt.colorbar(label="Real part of wavefunction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Wavefunction Real Part")
    plt.tight_layout()
    plt.show()