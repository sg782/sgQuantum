# Sullivan Gleason 
# March 13, 2025

import cmath

# we will assume the particle is massless
# therefore: omega = k * c, where k is a wave vector and c is the speed of light
#



def psi(r,t):
    # psi(r,t) = A * e ^ (k * r - w * t)
    #          = A * cos(k*r - w*t) + A * i * sin (k*r - w*t)

    A = 1.
    k = 1.
    
    