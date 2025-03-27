# 
# Following a split step fourier derivation for evolution of a quantum wavefunction, linked below:
# https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.03%3A_The_Split-Step_Fourier_Method
# 

import numpy as np
from wave_simulation.gaussian_wave import gaussian_wave_func_1d, gaussian_wave_func_2d


def initialize():

    # width = 
    subdivisions = 100


    X, Y = np.meshgrid(np.arange(subdivisions), np.arange(subdivisions))

    psi_0 = gaussian_wave_func_2d(0,0, t=0)

    print(psi_0)




    pass

def step_potential():
    pass

def step_kinetic():
    pass



initialize()