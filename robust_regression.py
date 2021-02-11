import numpy as np
from scipy.optimize import least_squares


# Uses a robust method of least squares regression.
# Returns coefficients
def parabola_fit(y):
    x = np.arange(len(y))
    mask = x > -1

    y_train = y[mask]
    t_train = x[mask]
    x0 = np.array([1.0, 1.0, 0.0])

    res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))

    return res_soft_l1.x


def fun(x, t, y):
    # return x[0] + x[1] * np.exp(x[2] * t) - y
    return x[0] * t ** 2 + x[1] * t + x[2] - y
