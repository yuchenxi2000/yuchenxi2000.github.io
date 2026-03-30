from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt

"""
solve poisson equation, on M x N square
boundary constraints:
v(y = N) = 1
v(y = -1) = 0
v(x = M) = 0
v(x = -1) = 0
"""

# shape
M = 100
N = 100


def is_boundary(x: int, y: int) -> (bool, float):
    """
    boundary constraints

    :param x:
    :param y:
    :return: Tuple, is on boundary & function value
    """
    if 0 <= x < M and 0 <= y < N:
        return False, 0
    elif y == N:
        return True, 1
    else:
        return True, 0


# construct A, b

data = np.zeros(M*N*5)
i_data = np.zeros(M*N*5)
j_data = np.zeros(M*N*5)

b = np.zeros(M*N)

i_x = np.zeros((M, N))
k = 0
for i in range(M):
    for j in range(N):
        i_x[i][j] = k
        k += 1

k = 0
n_eq = 0
for i in range(M):
    for j in range(N):
        # mid
        data[k] = 4
        i_data[k] = n_eq
        j_data[k] = i_x[i][j]
        k += 1
        # left
        boundary, value = is_boundary(i - 1, j)
        if boundary:
            b[n_eq] += value
        else:
            data[k] = -1
            i_data[k] = n_eq
            j_data[k] = i_x[i - 1][j]
            k += 1
        # right
        boundary, value = is_boundary(i + 1, j)
        if boundary:
            b[n_eq] += value
        else:
            data[k] = -1
            i_data[k] = n_eq
            j_data[k] = i_x[i + 1][j]
            k += 1
        # down
        boundary, value = is_boundary(i, j - 1)
        if boundary:
            b[n_eq] += value
        else:
            data[k] = -1
            i_data[k] = n_eq
            j_data[k] = i_x[i][j - 1]
            k += 1
        # up
        boundary, value = is_boundary(i, j + 1)
        if boundary:
            b[n_eq] += value
        else:
            data[k] = -1
            i_data[k] = n_eq
            j_data[k] = i_x[i][j + 1]
            k += 1
        n_eq += 1
data.resize(k)
i_data.resize(k)
j_data.resize(k)

A = csc_matrix((data, (i_data, j_data)), shape=(M*N, M*N))

v = spsolve(A, b)

v = v.reshape((M, N))

X = np.arange(0, M, 1)
Y = np.arange(0, N, 1)

X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, v, cmap=plt.cm.hot)
plt.show()
