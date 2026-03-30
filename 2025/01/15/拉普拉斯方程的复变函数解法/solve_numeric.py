from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

# shape
Nx = 200
Ny = 200
M = Nx - 1
N = Ny - 1

# construct A, b
data = np.zeros(M*N*5)
neq_data = np.zeros(M*N*5)
idx_data = np.zeros(M*N*5)
b = np.zeros(M*N)

ndata = 0
neq = 0
for i in range(M):
    for j in range(N):
        # mid
        data[ndata] = 4.0
        neq_data[ndata] = neq
        idx_data[ndata] = N * i + j
        ndata += 1
        # left
        if i == 0:
            b[neq] += 1.0
        else:
            data[ndata] = -1.0
            neq_data[ndata] = neq
            idx_data[ndata] = N * (i - 1) + j
            ndata += 1
        # right
        if i == M - 1:
            ...
        else:
            data[ndata] = -1.0
            neq_data[ndata] = neq
            idx_data[ndata] = N * (i + 1) + j
            ndata += 1
        # down
        if j == 0:
            ...
        else:
            data[ndata] = -1.0
            neq_data[ndata] = neq
            idx_data[ndata] = N * i + j - 1
            ndata += 1
        # up
        if j == N - 1:
            ...
        else:
            data[ndata] = -1
            neq_data[ndata] = neq
            idx_data[ndata] = N * i + j + 1
            ndata += 1
        neq += 1
data.resize(ndata)
neq_data.resize(ndata)
idx_data.resize(ndata)

A = csc_matrix((data, (neq_data, idx_data)), shape=(M*N, M*N))
v = spsolve(A, b)
v = v.reshape((M, N))

x = np.linspace(0, 1, num=Nx+1)
y = np.linspace(0, 1, num=Ny+1)
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

np.save('./solution/X.npy', X)
np.save('./solution/Y.npy', Y)
np.save('./solution/Z_numeric.npy', v)
