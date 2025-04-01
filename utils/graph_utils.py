import matplotlib.pyplot as plt

def initialize_plot(set_running):
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("2D Wavefunction probability density")
    plt.tight_layout()

    def on_key(e):
        global running
        global showing_3d

        if(e.key=="escape"):
            print("ESC pressed, stopping runtime")
            set_running(False)
            # running = False

    fig.canvas.mpl_connect('key_press_event', on_key)

    return fig, ax1

def update_plot(im, data):
    im.set_data(data)
    im.set_clim(vmin=data.min(), vmax=data.max())
