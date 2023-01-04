import numpy as np


def newton_method_with_armijo():
    pass


def newton_method_without_armijo():
    pass


def armijo_line_search():
    pass


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


if __name__ == "__main__":
    x0 = (-3/4, 1)
    print(compute_second_order_derivative(x0))





