import numpy as np
from scipy.optimize import least_squares


# Uses a robust method of least squares regression.
# Returns coefficients
def parabola_fit(y):
    y = np.asarray(y)
    x = np.arange(len(y))
    mask = x > -1

    y_train = y[mask]
    t_train = x[mask]
    x0 = np.array([1.0, 1.0, 0.0])

    res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t_train, y_train))

    # import matplotlib.pyplot as plt
    #
    # def gen_data(t, a, b, c, ):
    #     # return a + b * np.exp(t * c)
    #     return a * t ** 2 + b * t + c
    #
    # t_test = np.linspace(0, len(x), len(x) * 10)
    #
    # y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
    #
    # plt.plot(t_train, y_train, 'o')
    # plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
    # plt.xlabel("t")
    # plt.ylabel("y")
    #
    # plt.legend()
    # plt.show()

    return res_soft_l1.x


def fun(x, t, y):
    # return x[0] + x[1] * np.exp(x[2] * t) - y
    return x[0] * t ** 2 + x[1] * t + x[2] - y
