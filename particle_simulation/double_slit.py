# 
# Following a split step fourier derivation for evolution of a quantum wavefunction, linked below:
# https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.03%3A_The_Split-Step_Fourier_Method
# 

import numpy as np
import math
from wave_simulation.gaussian_wave import gaussian_wave_func_1d, gaussian_wave_func_2d, plot_wave_func_2d, plot_wave_func_3d
from fourier_transform import DFT_2d, IDFT_2d, batch_DFT_2d, batch_IDFT_2d
import matplotlib.pyplot as plt
import datetime
from utils.graph_utils import initialize_plot, update_plot

running = True

def initialize():
    width = 3 # units in space
    N = 70 # subsdivision
    dx = 2 * width / N

    X, Y = np.meshgrid(np.linspace(-width,width,N), np.linspace(-width,width,N))
    psi_0 = gaussian_wave_func_2d(X,Y, kx_0=20, ky_0=0, t=0, a=1)
    prob = abs(psi_0)**2

    potential_map = initialize_potential(N)
    
    plt.ion()


    fig, ax1 = initialize_plot(set_running)
    im1 = ax1.imshow(prob, origin='lower', cmap='viridis')
    cbar = fig.colorbar(im1, ax=ax1, label="probability density of wavefunction")


    for i in range(10000):
        if(not running): break

        psi_0 = evolve_full_step(psi_0,potential_map,0.0001,dx)


        if(i%10==0):
            prob = np.abs(psi_0)**2
            update_plot(im1,prob)

            plt.pause(0.001)

def initialize_potential(N):
    wall_i = int((9.5 * N) // 10)

    slit_1_start = int((3 * N) / 10)
    slit_1_end = int((4 * N) / 10)
    slit_2_start = int((6 * N) / 10)
    slit_2_end = int((7 * N) / 10)

    # create wall of potential energy
    potential_map = np.zeros((N, N))

    inf_potential = 1e300


    for i in range(N-wall_i):
        potential_map[N-i-1, :] = inf_potential #math.inf
        potential_map[i, :] = inf_potential #math.inf
        potential_map[:,i] = inf_potential #math.inf

    potential_map[0:slit_1_start,wall_i-4] = inf_potential #math.inf
    potential_map[0:slit_1_start,wall_i-5] = inf_potential #math.inf
    potential_map[0:slit_1_start,wall_i-6] = inf_potential #math.inf


    # 'door' to close slit
    # potential_map[slit_1_start:slit_1_end,wall_i-4] = inf_potential #math.inf
    # potential_map[slit_1_start:slit_1_end,wall_i-5] = inf_potential #math.inf
    # potential_map[slit_1_start:slit_1_end,wall_i-6] = inf_potential #math.inf
        
    potential_map[slit_1_end:slit_2_start,wall_i-4] = inf_potential #math.inf
    potential_map[slit_1_end:slit_2_start,wall_i-5] = inf_potential #math.inf
    potential_map[slit_1_end:slit_2_start,wall_i-6] = inf_potential #math.inf

    # ' door to close slit
    # potential_map[slit_2_start:slit_2_end,wall_i-4] = inf_potential #math.inf
    # potential_map[slit_2_start:slit_2_end,wall_i-5] = inf_potential #math.inf
    # potential_map[slit_2_start:slit_2_end,wall_i-6] = inf_potential #math.inf

    potential_map[slit_2_end:N-1,wall_i-4] = inf_potential #math.inf
    potential_map[slit_2_end:N-1,wall_i-5] = inf_potential #math.inf
    potential_map[slit_2_end:N-1,wall_i-6] = inf_potential #math.inf

    plot_wave_func_2d(potential_map)

    return potential_map

def set_running(new_running):
    # to handle escape event
    global running
    running = new_running

def step_potential(wave, potential, step):
    # since the potential energy is contstant throughout the simulation, the time component does not matter
    operator = np.exp(-1j * step * potential )

    wave_out = operator * wave
    return wave_out


def step_kinetic(wave,step, dx):
    
    N1, N2 = wave.shape

    checkerboard = (-1.) ** (np.indices(wave.shape).sum(axis=0))

    n1 = np.arange(N1)
    n2 = np.arange(N2)

    # kx = - math.pi / dx + (2 * math.pi * n1) / (dx * N1)
    # ky = - math.pi / dx + (2 * math.pi * n2) / (dx * N2)

    # # k_squared = np.outer(kx**2,ky**2)
    # k_squared = (kx**2) + (ky**2)

    
    kx = - math.pi / dx + (2 * math.pi * n1) / (dx * N1)
    ky = - math.pi / dx + (2 * math.pi * n2) / (dx * N2)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_squared = KX**2 + KY**2

    wave *= checkerboard # for (-1)^m

    wave_k = batch_DFT_2d(wave)

    coeff = np.exp((-1j*step/4)*k_squared)

    wave_k *= coeff

    inv_wave = batch_IDFT_2d(wave_k) * checkerboard

    return inv_wave

def evolve_full_step(wave,potential,dt,dx):
    wave = step_kinetic(wave,dt,dx)
    wave = step_potential(wave,potential,dt)
    wave = step_kinetic(wave,dt,dx)

    return wave

initialize()