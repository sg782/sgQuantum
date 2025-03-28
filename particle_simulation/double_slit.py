# 
# Following a split step fourier derivation for evolution of a quantum wavefunction, linked below:
# https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.03%3A_The_Split-Step_Fourier_Method
# 

import numpy as np
import math
from wave_simulation.gaussian_wave import gaussian_wave_func_1d, gaussian_wave_func_2d, plot_wave_func_2d
from fourier_transform import DFT_2d, IDFT_2d


def initialize():

    width = 10
    N = 10

    dx = 2 * width / N


    X, Y = np.meshgrid(np.linspace(-width,width,N), np.linspace(-width,width,N))

    # create wall of potential energy
    potential_map = np.zeros((N, N))
    # potential_map[3, :] = 1e10 #math.inf
    # potential_map[7, :] = 1e10 #math.inf
    potential_map[:,3] = 1e10 #math.inf
    potential_map[:,7] = 1e10 #math.inf




    # potential_map[30, 48] = 0
    # potential_map[30, 52] = 0

    psi_0 = gaussian_wave_func_2d(X,Y, kx_0=0, ky_0=10, t=0)

    plot_wave_func_2d(potential_map)
    plot_wave_func_2d(psi_0)

    for i in range(100):

        psi_0 = evolve_full_step(psi_0,potential_map,0.1,dx)
        if(i%10==0):
            plot_wave_func_2d(psi_0)



def step_potential(wave, potential, step):
    # since the potential energy is contstant throughout the simulation, the time component does not matter
    operator = np.exp(-1j * step * potential )

    wave_out = operator * wave
    return wave_out

def step_kinetic(wave,step, dx):
    N1, N2 = wave.shape

    # can vectorize
    n1 = np.arange(N1)
    n2 = np.arange(N2)

    kx = - math.pi / dx + (2 * math.pi * n1) / (dx * N1)
    ky = - math.pi / dx + (2 * math.pi * n2) / (dx * N2)

    k_squared = np.outer(kx**2,ky**2)

    for i in range(N1):
        for j in range(N2):
            wave[i][j] *= (-1)**(i+j)
    
    wave_k = DFT_2d(wave)

    coeff = np.exp((-1j*step/4)*k_squared)

    wave_k *= coeff

    inv_wave = IDFT_2d(wave_k)

    for i in range(N1):
        for j in range(N2):
            inv_wave[i][j] *= (-1)**(i+j)

    return inv_wave

def evolve_full_step(wave,potential,dt,dx):
    wave = step_kinetic(wave,dt,dx)
    print("Post k1 ", wave)
    wave = step_potential(wave,potential,dt)
    print("Post pot1 ", wave)


    wave = step_kinetic(wave,dt,dx)
    print("Post k2 ", wave)


    return wave





initialize()