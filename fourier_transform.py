# https://pycav.readthedocs.io/en/latest/api/pde/split_step.html
# 

import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
from matplotlib.widgets import Slider



def DFT_1d(data):
    out = np.zeros_like(data,dtype=np.complex128)
    N = len(data)

    for i in range(N):
        out[i] = np.sum(data * np.exp(-1j * 2 * cmath.pi * (i/N) * np.arange(N)))

    return out

def IDFT_1d(data):
    out = np.zeros_like(data,dtype=np.complex128)
    N = len(data)

    for i in range(N):
        out[i] = np.sum(data*np.exp(1j * 2 * cmath.pi * (i/N) * np.arange(N)))
        out[i] /= N

    return out


def DFT_2d(data):
    temp = np.zeros_like(data,dtype=np.complex128)
    out = np.zeros_like(data,dtype=np.complex128)
    N1, N2 = out.shape


    for i in range(N1):
        temp[i,:] = DFT_1d(data[i, :])
    
    for j in range(N2):
        out[:,j] = DFT_1d(temp[:,j])

    return out


def IDFT_2d(data):
    temp = np.zeros_like(data,dtype=np.complex128)
    out = np.zeros_like(data,dtype=np.complex128)
    N1, N2 = out.shape


    for i in range(N1):
        temp[i,:] = IDFT_1d(data[i, :])
    
    for j in range(N2):
        out[:,j] = IDFT_1d(temp[:,j])

    return out


def test():
    # 
    # Visualization with sliders thanks to the help of chatgpt (and myself of course)
    #
    #
    N = 300
    x = np.linspace(0, 10, N)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.31)

    (line_orig,) = ax1.plot(x, np.zeros_like(x), label="Original f(x)")
    ax1.set_title("Original Signal")
    ax1.set_ylim(-3, 3)
    ax1.legend()

    (line_dft,) = ax2.plot(x, np.zeros_like(x), label="DFT Magnitude")
    ax2.set_title("DFT Magnitude Spectrum")
    ax2.set_ylim(0, 5)
    ax2.legend()

    (line_recon,) = ax3.plot(x, np.zeros_like(x), label="Reconstructed Signal")
    ax3.set_title("Reconstructed Signal (IDFT)")
    ax3.set_ylim(-3, 3)
    ax3.legend()

    ax_alpha = plt.axes([0.1, 0.22, 0.8, 0.03])
    ax_beta = plt.axes([0.1, 0.17, 0.8, 0.03])
    ax_m1 = plt.axes([0.1, 0.12, 0.8, 0.03])
    ax_m2 = plt.axes([0.1, 0.07, 0.8, 0.03])

    slider_alpha = Slider(ax_alpha, 'Alpha', 0.1, 10.0, valinit=1.0)
    slider_beta = Slider(ax_beta, 'Beta', 0.1, 10.0, valinit=2.0)

    slider_magnitude_1 = Slider(ax_m1, 'Magnitude 1', 0.1, 10.0, valinit=1.0)
    slider_magnitude_2 = Slider(ax_m2, 'Magnitude 2', 0.1, 10.0, valinit=2.0)

    def update(val):
        alpha = slider_alpha.val
        beta = slider_beta.val
        m1 = slider_magnitude_1.val
        m2 = slider_magnitude_2.val

        f_x = m1 * np.cos(2 * math.pi * alpha * x) + m2 * np.cos(2 * math.pi * beta * x)
        out = DFT(f_x)
        back = IDFT(out)

        line_orig.set_ydata(f_x)
        line_dft.set_ydata(np.abs(out) / N)
        line_recon.set_ydata(back.real)

        ax1.set_ylim(-abs(m1 + m2), m1+m2)
        ax3.set_ylim(-abs(m1 + m2), m1+m2)

        fig.canvas.draw_idle()

    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    slider_magnitude_1.on_changed(update)
    slider_magnitude_2.on_changed(update)

    update(None) 
    plt.show()

def test_2d():
    data = np.random.rand(10,10)

    dft_data = DFT_2d(data)

    inv_dft_data = IDFT_2d(dft_data)

    print((data-inv_dft_data).sum())



