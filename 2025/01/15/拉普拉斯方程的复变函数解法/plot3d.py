import numpy as np
import matplotlib.pyplot as plt

X = np.load('./solution/X.npy')
Y = np.load('./solution/Y.npy')
Z_analytic = np.load('./solution/Z_analytic.npy')
Z_numeric = np.load('./solution/Z_numeric.npy')

err = np.max(np.abs(Z_numeric - Z_analytic))
print(f'grid size: {X.shape[0] + 1} * {X.shape[1] + 1}')
print(f'diff between numeric & analytic: {err}')

fig = plt.figure()
ax3d = plt.axes(projection='3d')
plt.title('Solution of Laplace Equation')

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

ax3d.plot_surface(X, Y, Z_analytic, cmap='rainbow')
plt.show()
