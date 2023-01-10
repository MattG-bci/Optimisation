import matplotlib.pyplot as plt
import numpy as np

# Create a grid of points
X, Y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))

# Calculate the function values at each point
Z = -X*Y

mask = np.round(Z) == -1



# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(X[mask], Y[mask], Z[mask], c='blue', linewidth=6, alpha=1)


plt.show()