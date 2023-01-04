import numpy as np
import matplotlib.pyplot as plt


def gradient_method(x0, tolerance=1e-4, max_iter=1000):
    k = 0
    x_k = [x0]
    d_k = compute_gradient(x0)
    while np.all(abs(d_k) > tolerance) and k < max_iter:
        d_k = -compute_gradient(x_k[-1])

        # compute a step alpha k along a direction d_k with any line search method (use Armijo here)
        alpha = armijo_line_search(x_k[-1], d_k) # rosenbrock of the previous point for comparison
        x = x_k[-1] + alpha * d_k
        k += 1
        x_k.insert(k, x)

    return x_k

def armijo_line_search(x_k, d_k, sigma=1e-2, gamma=0.2, alpha=0.5):
    f_xk = compute_rosenbrock(x_k)
    f_xk_update = compute_rosenbrock(x_k + alpha * d_k)
    f_xk_gradient = compute_gradient(x_k)
    while np.all(f_xk_update > f_xk + gamma * alpha * f_xk_gradient * d_k):
        alpha = sigma * alpha
        f_xk = compute_rosenbrock(x_k)
        f_xk_update = compute_rosenbrock(x_k + alpha * d_k)
        f_xk_gradient = compute_gradient(x_k)
    return alpha




def compute_gradient(point):
    x = point[0]
    y = point[1]
    return np.array([-400*x*(y - x**2) - 2*(1 - x), 200*(y-x**2)])

def compute_rosenbrock(point):
    x = point[0]
    y = point[1]
    return 100*np.power((y - np.power(x, 2)), 2) + np.power((1-x), 2)

if __name__ == "__main__":
    x0 = (-3/4, 1)
    x_star = gradient_method(x0)
    print(x_star[-1])