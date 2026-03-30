"""
Calculate Green's function in 3D (video rendering)
Please refer to Fig. 6.1, Chapter 6, QUANTUM THEORY OF THE ELECTRON LIQUID
Constant (multiplier) is ignored for simplicity, which means physical quantities are scaled
Numerical integral is more numerically stable than analytical expression
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def f(x, r, t):
    return x * np.sin(r * x) / r * (np.cos(x ** 2 * t) - 1j * np.sin(x ** 2 * t))


def int_inf(r, t):
    return -np.sqrt(2.0 * np.pi) / 8.0 * (1.0+1.0j) * np.exp(1j * r ** 2 / 4 / t) * np.power(t, -1.5)


def G(r, t):
    def ft_real(x):
        return f(x, r, t).real
    def ft_imag(x):
        return f(x, r, t).imag
    q0 = scipy.integrate.quad(ft_real, 0.0, 1.0)
    q1 = scipy.integrate.quad(ft_imag, 0.0, 1.0)
    assert q0[1] < 1e-5 and q1[1] < 1e-5
    if t > 0:
        return 3.0 * (-q1[0] + 1j * q0[0] - 1j * int_inf(r, t))
    elif t < 0:
        return 3.0 * (-q1[0] + 1j * q0[0])
    else:
        return np.nan


def render_pics():
    plt.figure(figsize=(12, 6))
    for Nt in range(-360, 361):
        t = Nt / 24.0
        r_arr = []
        g_arr0 = []
        # g_arr1 = []
        for Nr in range(1, 800):
            r = Nr / 100
            g = G(r, t)
            r_arr.append(r)
            g_arr0.append(g.real)
            # g_arr1.append(g.imag)
        plt.cla()
        plt.plot(r_arr, g_arr0, '-r')
        # plt.plot(r_arr, g_arr1, '-b')
        plt.ylim(-1.5, +1.5)
        plt.title(f't = {t:+.2f}')
        plt.savefig(f'fig/{(Nt+360):03}.jpg')


def plot(t=1.0):
    r_arr = []
    g_arr0 = []
    g_arr1 = []
    for Nr in range(1, 800):
        r = Nr / 100
        g = G(r, t)
        r_arr.append(r)
        g_arr0.append(g.real)
        g_arr1.append(g.imag)
    plt.cla()
    plt.plot(r_arr, g_arr0, '-r')
    plt.plot(r_arr, g_arr1, '-b')
    plt.ylim(-5, +5)
    plt.title(f't = {t:+.2f}')
    plt.show()


render_pics()
