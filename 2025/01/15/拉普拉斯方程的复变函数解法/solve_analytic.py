import scipy
import numpy as np

Nx = 200
Ny = 200
x = np.linspace(0, 1, num=Nx+1)
y = np.linspace(0, 1, num=Ny+1)
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

m = 0.5
K = scipy.special.ellipk(m)


def u(x, y):
    sn_x, cn_x, dn_x, ph_x = scipy.special.ellipj(x*K, m)
    sn_y, cn_y, dn_y, ph_y = scipy.special.ellipj(y*K, m)
    return 2.0 / np.pi * np.arctan(dn_x * cn_x * cn_y * sn_y / (sn_x * dn_y))


Z = np.zeros_like(X)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = u(X[i, j], Y[i, j])

np.save('./solution/X.npy', X)
np.save('./solution/Y.npy', Y)
np.save('./solution/Z_analytic.npy', Z)
