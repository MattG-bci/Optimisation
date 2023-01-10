import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(point):
    x = point[0]
    y = point[1]
    return np.array([-400*x*(y - x**2) - 2*(1 - x), 200*(y-x**2)])


def compute_second_order_derivative(point):
    x = point[0]
    y = point[1]
    return np.array([[-400*x*(y - 3*x**2) + 2, -400*x], [-400*x, 200]])


def compute_rosenbrock(point):
    x = point[0]
    y = point[1]
    return 100*np.power((y - np.power(x, 2)), 2) + np.power((1-x), 2)


def armijo_line_search(x_k, d_k, sigma=1e-1, gamma=0.8, alpha=100):
    f_xk = compute_rosenbrock(x_k)
    f_xk_update = compute_rosenbrock(x_k + alpha * d_k)
    f_xk_gradient = compute_gradient(x_k)
    while np.all(f_xk_update > f_xk + gamma * alpha * f_xk_gradient.dot(d_k)):
        alpha = sigma * alpha
        f_xk = compute_rosenbrock(x_k)
        f_xk_update = compute_rosenbrock(x_k + alpha * d_k)
        f_xk_gradient = compute_gradient(x_k)
    return alpha


def compute_polak_riberie_method(x0, max_iter=1000):
    x_k = x0
    k = 0
    d_k = compute_gradient(x0)
    x = [x0]
    costs = []

    while np.all(d_k != 0) and k < max_iter:
        if k == 0:
            d_k = -d_k
        else:
            gradient = compute_gradient(x_k)
            d_k = -gradient + (gradient.dot(gradient - d_k)/(np.linalg.norm(d_k))**2) * d_k

        
        Jk = cost(x_k)
        costs.append(Jk)
        alpha = armijo_line_search(x_k, d_k)
        x_k = x_k + alpha * d_k
        x.append(x_k)
        k += 1

    return x, costs


def cost(point):
    x = point[0]
    y = point[1]
    return np.log((x - 1)**2 + (y - 1)**2)


if __name__ == "__main__":
    x0 = (-0.75, 1)
    x_star, costs = compute_polak_riberie_method(x0)
    print(f"Optimal point using Polak-Riberie algorithm with Armijo line search: {x_star[-1]}")

    points = np.array(x_star)
    X = points[:, 0]
    Y = points[:, 1]

    plt.plot(X, Y)
    plt.scatter(-0.75, 1, color="green", label="Initial point")
    plt.scatter(1, 1, color="red", label="Stationary point of v(x,y)")
    plt.scatter(X[-1], Y[-1], color="purple", label="Converged point")
    plt.title("Polak-Riberie algorithm with Armijo line search after 1000 iterations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.savefig("polak_riberie_method_plot.jpg")
    plt.show()

    plt.plot(range(0, 1000), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Convergence of the Polak-Riberie algorithm with Armijo line search")
    plt.grid()
    plt.savefig("polak_riberie_method_convergence.jpg")
    plt.show()


