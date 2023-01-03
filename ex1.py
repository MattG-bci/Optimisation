import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Defining the Rosenbrock function and range of values for the inputs x and y.
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    v = 100*np.power((y - np.power(x, 2)), 2) + np.power((1-x), 2)

    # plot the function on the 3D plane
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection="3d")
    ax.plot3D(x, y, v, 'black')
    ax.plot3D(x, y, 4000, "red", label="level set = 4000")
    ax.plot3D(x, y, 2000, "blue", label="level set = 2000")
    ax.plot3D(x, y, 1000, "green", label="level set = 1000")
    ax.plot3D(x, y, 500, "orange", label="level set = 500")
    ax.set_title("Rosenbrock function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="center")

    # show the plot
    plt.show()

    # saving the plot
    fig.savefig("Level sets Rosenbrock")
