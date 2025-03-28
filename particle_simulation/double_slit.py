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

running = True

def initialize():

    width = 3
    N = 200

    dx = 2 * width / N

    wall_i = 8 * N // 10

    inf_potential = 1e300


    X, Y = np.meshgrid(np.linspace(-width,width,N), np.linspace(-width,width,N))

    # create wall of potential energy
    potential_map = np.zeros((N, N))
    potential_map[N-wall_i, :] = inf_potential #math.inf
    potential_map[wall_i, :] = inf_potential #math.inf
    potential_map[:,N-wall_i] = inf_potential #math.inf
    potential_map[:,wall_i] = inf_potential #math.inf

    #outer walls
    potential_map[0,:] = inf_potential
    potential_map[N-1,:] = inf_potential
    potential_map[:,0] = inf_potential
    potential_map[:,N-1] = inf_potential

    potential_map[N//2+8,wall_i] = 0 
    potential_map[N//2+7,wall_i] = 0 
    potential_map[N//2+6,wall_i] = 0 
    potential_map[N//2+5,wall_i] = 0 
    potential_map[N//2+4,wall_i] = 0 
    potential_map[N//2+3,wall_i] = 0 

    potential_map[N//2-3,wall_i] = 0 
    potential_map[N//2-4,wall_i] = 0 
    potential_map[N//2-5,wall_i] = 0 
    potential_map[N//2-6,wall_i] = 0 
    potential_map[N//2-7,wall_i] = 0 
    potential_map[N//2-8,wall_i] = 0 


    potential_map[0,:] = inf_potential

    # potential_map[30, 48] = 0
    # potential_map[30, 52] = 0

    psi_0 = gaussian_wave_func_2d(X,Y, kx_0=10, ky_0=0, t=0)

    plot_wave_func_2d(potential_map)

    prob = abs(psi_0)**2
    
    plt.ion()

    # plot_dimensions = 2

    # fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(6, 5))
    fig, ax1 = plt.subplots(figsize=(6, 5))
    im1 = ax1.imshow(prob, origin='lower', cmap='viridis')
    cbar = fig.colorbar(im1, ax=ax1, label="probability density of wavefunction")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("2D Wavefunction probability density")
    plt.tight_layout()


    def on_key(e):
        global running
        global showing_3d

        if(e.key=="escape"):
            print("ESC pressed, stopping runtime")
            running = False

    fig.canvas.mpl_connect('key_press_event', on_key)

    for i in range(10000):
        if(not running): break

        psi_0 = evolve_full_step(psi_0,potential_map,0.0001,dx)

        if(i%10==0):
            # if(not showing_3d):
            prob = np.abs(psi_0)**2
            im1.set_data(prob)
            im1.set_clim(vmin=prob.min(), vmax=prob.max())

            # im2.set_data(psi_0.real)
            # im2.set_clim(vmin=psi_0.real.min(), vmax=psi_0.real.max())
            plt.pause(0.001)


def step_potential(wave, potential, step):
    # since the potential energy is contstant throughout the simulation, the time component does not matter
    operator = np.exp(-1j * step * potential )

    wave_out = operator * wave
    return wave_out


def step_kinetic(wave,step, dx):
    
    start = datetime.datetime.now()
    

    N1, N2 = wave.shape

    checkerboard = (-1.) ** (np.indices(wave.shape).sum(axis=0))

    n1 = np.arange(N1)
    n2 = np.arange(N2)

    kx = - math.pi / dx + (2 * math.pi * n1) / (dx * N1)
    ky = - math.pi / dx + (2 * math.pi * n2) / (dx * N2)

    k_squared = np.outer(kx**2,ky**2)

    wave *= checkerboard # for (-1)^m




    wave_k = batch_DFT_2d(wave)

    end = datetime.datetime.now()
    # print("Small section: ", end-start)

    coeff = np.exp((-1j*step/4)*k_squared)

    wave_k *= coeff

    inv_wave = batch_IDFT_2d(wave_k)

    inv_wave *= checkerboard

    return inv_wave

def evolve_full_step(wave,potential,dt,dx):

    start = datetime.datetime.now()
    wave = step_kinetic(wave,dt,dx)

    end = datetime.datetime.now()
    print("Time elapsed kinetic 1: ", end-start)


    wave = step_potential(wave,potential,dt)

    wave = step_kinetic(wave,dt,dx)



    return wave






initialize()