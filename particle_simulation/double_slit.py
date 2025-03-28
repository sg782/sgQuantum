# 
# Following a split step fourier derivation for evolution of a quantum wavefunction, linked below:
# https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.03%3A_The_Split-Step_Fourier_Method
# 

import numpy as np
import math
from wave_simulation.gaussian_wave import gaussian_wave_func_1d, gaussian_wave_func_2d, plot_wave_func_2d
from fourier_transform import DFT_2d, IDFT_2d, batch_DFT_2d, batch_IDFT_2d
import matplotlib.pyplot as plt
import datetime

running = True


def on_key(e):
    global running
    if(e.key=="escape"):
        print("ESC pressed, stopping runtime")
        running = False

def initialize():

    width = 3
    N = 120

    dx = 2 * width / N

    wall_i = 8 * N // 10

    inf_potential = 1e100


    X, Y = np.meshgrid(np.linspace(-width,width,N), np.linspace(-width,width,N))

    # create wall of potential energy
    potential_map = np.zeros((N, N))
    # potential_map[N-wall_i, :] = 1e100 #math.inf
    # potential_map[wall_i, :] = 1e100 #math.inf
    # potential_map[:,N-wall_i] = 1e100 #math.inf
    # potential_map[:,wall_i] = 1e100 #math.inf

    # potential_map[N//2+4,wall_i] = 0 #math.inf
    # potential_map[N//2+3,wall_i] = 0 #math.inf
    # potential_map[N//2+2,wall_i] = 0 #math.inf

    # potential_map[N//2-2,wall_i] = 0 #math.inf
    # potential_map[N//2-3,wall_i] = 0 #math.inf
    # potential_map[N//2-4,wall_i] = 0 #math.inf

    # potential_map[30, 48] = 0
    # potential_map[30, 52] = 0

    psi_0 = gaussian_wave_func_2d(X,Y, kx_0=0, ky_0=0, t=0)

    plot_wave_func_2d(potential_map)

    prob = abs(psi_0)**2
    
    plt.ion()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(psi_0.real, origin='lower', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, label="probability density of wavefunction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Wavefunction probability density")
    plt.tight_layout()

    fig.canvas.mpl_connect('key_press_event', on_key)


    for i in range(1000):
        if(not running): break

        psi_0 = evolve_full_step(psi_0,potential_map,0.1,dx)


        prob = np.abs(psi_0)**2
        im.set_data(psi_0.real)
        im.set_clim(vmin=prob.min(), vmax=prob.max())
        plt.pause(0.01)



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

    kx = - math.pi / dx + (2 * math.pi * n1) / (dx * N1)
    ky = - math.pi / dx + (2 * math.pi * n2) / (dx * N2)

    k_squared = np.outer(kx**2,ky**2)

    wave *= checkerboard # for (-1)^m

    wave_k = batch_DFT_2d(wave)

    coeff = np.exp((-1j*step/4)*k_squared)

    wave_k *= coeff

    inv_wave = batch_IDFT_2d(wave_k)

    inv_wave *= checkerboard

    return inv_wave

def evolve_full_step(wave,potential,dt,dx):

    start = datetime.datetime.now()
    wave = step_kinetic(wave,dt,dx)

    end = datetime.datetime.now()
    print("Time elapsed: ", end-start)


    wave = step_potential(wave,potential,dt)


    wave = step_kinetic(wave,dt,dx)



    return wave






initialize()