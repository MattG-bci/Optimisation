import numpy as np
import matplotlib.pyplot as plt

def armijo_line_search(x_k, d_k, sigma=1e-1, gamma=0.8, alpha=0.5):
    f_xk = compute_rosenbrock(x_k)
    f_xk_update = compute_rosenbrock(x_k + alpha * d_k)
    f_xk_gradient = compute_gradient(x_k)
    while np.all(f_xk_update > f_xk + gamma * alpha * f_xk_gradient.dot(d_k)):
        alpha = sigma * alpha
        f_xk = compute_rosenbrock(x_k)
        f_xk_update = compute_rosenbrock(x_k + alpha * d_k)
        f_xk_gradient = compute_gradient(x_k)
    return alpha


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


def newton_method_with_armijo(x0, alpha=0.5, gamma=0.8, max_iter=1000, tol=1e-6):
    x_k = x0
    costs = []
    x = [x0]
    for k in range(max_iter):
        grad = compute_gradient(x_k)
        hessian = compute_second_order_derivative(x_k)
        direction = -np.linalg.solve(hessian, grad)
        Jk = cost(x_k)
        costs.append(Jk)
        
        # Armijo line search
        alpha = armijo_line_search(x_k, direction)
        x_k = x_k + alpha * direction
        x.append(x_k)

        if np.linalg.norm(grad) < tol:
            break

    return x, costs


def newton_method_without_armijo(x_0, alpha=0.5, beta=0.8, max_iter=1000, tol=1e-6):
    x_k = x0
    costs = []
    x = [x0]
    for i in range(max_iter):
        grad = compute_gradient(x_k)
        hessian = compute_second_order_derivative(x_k)
        direction = -np.linalg.solve(hessian, grad)
        Jk = cost(x_k)
        costs.append(Jk)
        
        x_k = x_k + alpha * direction
        x.append(x_k)
        
        if np.linalg.norm(grad) < tol:
            break
    return x, costs


def cost(point):
    x = point[0]
    y = point[1]
    return np.log((x - 1)**2 + (y - 1)**2)


if __name__ == "__main__":
    x0 = (-0.75, 1)
    x_star, costs = newton_method_with_armijo(x0)
    print(f"Newton method with Armijo line search optimal point: {x_star[-1]}")
    points = np.array(x_star)
    X = points[:, 0]
    Y = points[:, 1]

    plt.plot(X, Y)
    plt.scatter(-0.75, 1, color="green", label="Initial point")
    plt.scatter(1, 1, color="red", label="Stationary point of v(x,y)")
    plt.scatter(X[-1], Y[-1], color="purple", label="Converged point")
    plt.title("Newton's method with Armijo line search after 1000 iterations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.savefig("newton_armijo_method_plot.jpg")
    plt.show()

    plt.plot(range(0, 1000), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Convergence of the Newton's method with Armijo line search")
    plt.grid()
    plt.savefig("newton_armijo_method_convergence.jpg")
    plt.show()


    x_star, costs = newton_method_without_armijo(x0)
    print(f"Newton method without Armijo line search optimal point: {x_star[-1]}")
    points = np.array(x_star)
    X = points[:, 0]
    Y = points[:, 1]

    plt.plot(X, Y)
    plt.scatter(-0.75, 1, color="green", label="Initial point")
    plt.scatter(1, 1, color="red", label="Stationary point of v(x,y)")
    plt.scatter(X[-1], Y[-1], color="purple", label="Converged point")
    plt.title("Newton's method without Armijo line search after 1000 iterations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.savefig("newton_method_plot.jpg")
    plt.show()

    plt.plot(range(0, 1000), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Convergence of the Newton's method without Armijo line search")
    plt.grid()
    plt.savefig("newton_method_convergence.jpg")
    plt.show()
