import numpy as np
import matplotlib.pyplot as plt


def gradient_method(x0):
    k = 0
    gradient = compute_gradient(x0)
    x_k = [x0]
    d_k = []
    alpha = 0.5
    while np.all(gradient != 0):
        d_k = -compute_gradient(x_k[-1])

        # compute a step alpha k along a direction d_k with any line search method (use Armijo here)
        x_prev = compute_rosenbrock(x_k[-1]) # rosenbrock of the previous point for comparison
        x = x_k[-1] + alpha*d_k
        k += 1
        x_k.insert(k, x)
        print(compute_rosenbrock(x_k[-1]))
        print(x_k)

        break

    return x_k

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