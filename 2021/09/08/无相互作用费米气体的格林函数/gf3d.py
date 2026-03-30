"""
Calculate Green's function in 3D at r = 0
Please refer to Fig. 6.1, Chapter 6, QUANTUM THEORY OF THE ELECTRON LIQUID
Constant (multiplier) is ignored for simplicity, which means physical quantities are scaled
Numerical integral is more numerically stable than analytical expression
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def f(x, t):
    return x ** 2 * (np.cos(x ** 2 * t) - 1j * np.sin(x ** 2 * t))


def int_inf(t):
    return -np.sqrt(2.0 * np.pi) / 8.0 * (1.0 + 1.0j) * np.power(t, -1.5)


def G(t):
    def ft_real(x):
        return f(x, t).real
    def ft_imag(x):
        return f(x, t).imag
    q0 = scipy.integrate.quad(ft_real, 0.0, 1.0)
    q1 = scipy.integrate.quad(ft_imag, 0.0, 1.0)
    if t > 0:
        return 3.0 * (-q1[0] + 1j * q0[0] - 1j * int_inf(t))
    elif t < 0:
        return 3.0 * (-q1[0] + 1j * q0[0])
    else:
        return np.nan


tarray = []
garray0 = []
garray1 = []


for Nt in range(-1500, +1500):
    if Nt == 0:
        continue
    t = Nt / 100
    g = G(t)
    tarray.append(t)
    if np.abs(g.real) > 1.2:
        garray0.append(np.nan)
    else:
        garray0.append(g.real)
    if np.abs(g.imag) > 1.2:
        garray1.append(np.nan)
    else:
        garray1.append(g.imag)

plt.plot(tarray, garray0, '-r')
plt.plot(tarray, garray1, '-b')
plt.xlabel('t')
plt.ylabel('$G(r=0, t)$')
plt.title('Green\'s Function in 3D at r = 0')
plt.ylim(-1.2, 1.2)
plt.show()
