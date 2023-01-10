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


def compute_broyden_fletcher_algorithm(x0, max_iter=98):
    x_k = [x0]
    k = 0
    gradient = compute_gradient(x0)
    costs = []

    while np.all(gradient != 0) and k < max_iter:
        if k == 0:
            H = np.identity(len(x0))
        else:
            delta = x_k[-1] - x_k[-2]
            gamma = compute_gradient(x_k[-1]) - compute_gradient(x_k[-2])

            if np.all(gamma == 0) or np.all(delta == 0):
                break

            v_k = np.power(np.dot(gamma.T, (np.dot(H, gamma))), 1/2) * (delta/(np.dot(delta.T, gamma))) - np.dot(H, gamma)/(np.dot(gamma.T, np.dot(H, gamma)))
            H = H + delta.dot(delta)/(np.dot(delta.T, gamma)) - (np.dot(H, np.dot(gamma, np.dot(gamma.T, H))))/(np.dot(gamma.T, np.dot(H, gamma))) + np.dot(v_k, v_k.T)

        gradient = compute_gradient(x_k[-1])
        d_k = -np.dot(H, gradient)
        alpha = armijo_line_search(x_k[-1], d_k)
        x = x_k[-1] + alpha * d_k
        Jk = cost(x)
        costs.append(Jk)
        x_k.append(x)
        k += 1
    return x_k, costs

def cost(point):
    x = point[0]
    y = point[1]
    return np.log((x - 1)**2 + (y - 1)**2)


if __name__ == "__main__":
    x0 = (-0.75, 1)
    x_star, costs = compute_broyden_fletcher_algorithm(x0)
    print(f"Optimal point using Broyden-Fletcher-Goldfarb-Shanno algorithm with Armijo line search: {x_star[-1]}")

    points = np.array(x_star)
    X = points[:, 0]
    Y = points[:, 1]

    plt.plot(X, Y)
    plt.scatter(-0.75, 1, color="green", label="Initial point")
    plt.scatter(1, 1, color="red", label="Stationary point of v(x,y)")
    plt.scatter(X[-1], Y[-1], color="purple", label="Converged point")
    plt.title("BFGS algorithm with Armijo line search after 1000 iterations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.savefig("bfgs_method_plot.jpg")
    plt.show()

    plt.plot(range(0, 98), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Convergence of the BFGS algorithm with Armijo line search")
    plt.grid()
    plt.savefig("bfgs_method_convergence.jpg")
    plt.show()
