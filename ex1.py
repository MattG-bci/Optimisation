from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == "__main__":
    # Defining the Rosenbrock function and range of values for the inputs x and y.
    x = np.linspace(-4, 4, 8000)
    y = np.linspace(-4, 4, 8000)
    v = 100*np.power((y - np.power(x, 2)), 2) + np.power((1-y), 2)
    X, Y = np.meshgrid(x, y)
    V = 100*np.power((Y - np.power(X, 2)), 2) + np.power((1-Y), 2)

    mask1 = np.round(V) == 20000
    mask2 = np.round(V) == 10000
    mask3 = np.round(V) == 5000
    mask4 = np.round(V) == 1000
    # plot the function on the 3D plane
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, V, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
    ax.scatter(X[mask1], Y[mask1], V[mask1], c='blue', linewidth=6, alpha=1, label="Level set = 20000")
    ax.scatter(X[mask2], Y[mask2], V[mask2], c='green', linewidth=6, alpha=1, label="Level set = 10000")
    ax.scatter(X[mask3], Y[mask3], V[mask3], c='orange', linewidth=6, alpha=1, label="Level set = 5000")
    ax.scatter(X[mask4], Y[mask4], V[mask4], c='yellow', linewidth=6, alpha=1, label="Level set = 1000")
    ax.set_title("Rosenbrock function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("v(x, y)", labelpad=8)
    ax.legend(loc="upper left")

    # show the plot
    plt.show()

    # saving the plot
    fig.savefig("Level sets Rosenbrock")
