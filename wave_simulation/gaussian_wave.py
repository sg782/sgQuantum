import math
import cmath

def gaussian_wave_func_2d(x,y,t,a=1,m=1,h_bar=1,k_0=1):

    val_x = gaussian_wave_func_1d(x,t,a,m,h_bar,k_0)
    val_y = gaussian_wave_func_1d(y,t,a,m,h_bar,k_0)

    out = val_x * val_y
    return out

def gaussian_wave_func_1d(x,t,a=1,m=1,h_bar=1,k_0=1):

    theta = math.pi # Need to define theta

    phi = - theta - (t * (h_bar * (k_0 **2))/(2*m))


    p1 = (2 * (a**2)/math.pi) ** (1/4)

    p2_num = cmath.exp(1j*phi) * cmath.exp(1j*k_0*x)
    p2_den = ((a**4) + (4*(h_bar**2) * (t**2)) / (m**2))**(1/4)

    p2 = p2_num / p2_den

    p3_num = - (x - (h_bar* k_0*t)/m) **2
    p3_den = (a**2) + (2*1j * h_bar * t) / m

    p3 = cmath.exp(p3_num/p3_den)

    return p1 * p2 * p3

