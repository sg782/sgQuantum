#
# https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/07%3A_Quantum_Mechanics/7.02%3A_Wavefunctions#:~:text=The%20wavefunction%20of%20a%20light%20wave%20is%20given,2%20is%20proportional%20to%20the%20number%20of%20photons.
# https://en.wikipedia.org/wiki/Double-slit_experiment
# https://en.wikipedia.org/wiki/Huygens%E2%80%93Fresnel_principle
# https://en.wikipedia.org/wiki/Particle_in_a_box
# https://javalab.org/en/double_slit_en/
# https://pycav.readthedocs.io/en/latest/api/pde/split_step.html
# https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.03%3A_The_Split-Step_Fourier_Method
# https://en.wikipedia.org/wiki/Wave_packet#The_2D_case

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmath
from matplotlib.widgets import Slider
from wave_simulation.gaussian_wave import gaussian_wave_func_1d, gaussian_wave_func_2d





# grid_width = 25
# grid = np.array([0] * grid_width * grid_width).reshape((grid_width,grid_width))

# print(grid)


def wave(x,t):
    c = 10 # speed of propagation 
    lam = 1 # lambda
    coeff = math.exp(-(x-c*t)**2)
    argument = 2*math.pi* (x-c*t)/lam

    real = coeff * math.cos(argument)
    imag = coeff * math.sin(argument)

    return real, imag

def plot(t):
    width = 5*math.pi
    n = 100
    dx = width / n

    # t = 0

    reals = []
    imags = []
    pos = []

    values = np.zeros((2*n, 2*n))

    for i in range(-n,n):
        for j in range(-n,n):
            x = i * dx
            y = j * dx
            
            # val = gaussian_wave_func_2d(i,j)

            val = abs(gaussian_wave_func_2d(x, y,t)) ** 2  # assuming this returns a complex value
            values[i + n, j + n] = val
            
            # reals.append(val.real)
            # imags.append(val.imag)

    plt.figure(figsize=(6, 5))
    plt.imshow(values, extent=[-n*dx, n*dx, -n*dx, n*dx], origin='lower', cmap='viridis')
    plt.colorbar(label="Real part of wavefunction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Wavefunction Real Part")
    plt.tight_layout()
    plt.show()
    return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs, reals, imags, color='purple', linewidth=1)

    # Set axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("Real part")
    ax.set_zlabel("Imag part")
    ax.set_title("Wavefunction in 3D (Complex Plane)")

    plt.tight_layout()
    plt.show()

def calculate_map(t,m,h_bar,k_0):
    width = 5*math.pi
    n = 20
    dx = width / n
    values = np.zeros((2*n, 2*n))

    for i in range(-n,n):
        for j in range(-n,n):
            x = i * dx
            y = j * dx
            
            # val = gaussian_wave_func_2d(i,j)

            val = abs(gaussian_wave_func_2d(x, y,t,m,h_bar,k_0)) ** 2  # assuming this returns a complex value
            values[i + n, j + n] = val  # shift indices so they start at 0

    extent=[-n*dx, n*dx, -n*dx, n*dx]

    return values, extent


initial_vals, extent = calculate_map(0,1,1,1)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

heatmap = ax.imshow(initial_vals, extent=extent, origin='lower', cmap='viridis')
cbar = plt.colorbar(heatmap)
cbar.set_label("Amplitude")

def update(val):
    t = slider_t.val
    m = slider_m.val
    h_bar = slider_h_bar.val
    k_0 = slider_k_0.val

    vals, extent = calculate_map(t,m,h_bar,k_0)
    heatmap.set_data(vals)
    heatmap.set_clim(vals.min(),vals.max())  # update color scale
    fig.canvas.draw_idle()


ax_t = plt.axes([0.15, 0.25, 0.65, 0.03])
slider_t = Slider(ax_t, 't',0,10.0, valinit=0)
slider_t.on_changed(update)

ax_m = plt.axes([0.15, 0.2, 0.65, 0.03])
slider_m = Slider(ax_m, 'm',0.1,10.0, valinit=1)
slider_m.on_changed(update)

ax_h_bar = plt.axes([0.15, 0.15, 0.65, 0.03])
slider_h_bar = Slider(ax_h_bar, 'h_bar',0.1,10.0, valinit=1)
slider_h_bar.on_changed(update)

ax_k_0 = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_k_0 = Slider(ax_k_0, 'k_0',0.1,10.0, valinit=1)
slider_k_0.on_changed(update)



plt.title("Interactive 2D Gaussian Wavefunction")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


    
    