import numpy as np
from scipy.optimize import least_squares, minimize


# Uses a robust method of least squares regression.
# Returns coefficients
def parabola_fit(y):
    y = np.asarray(y)
    # y = np.nan_to_num(y, nan=-1)
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


# Returns two separate x and y variables containing the coordinates
# required for the piecewise linear regression lines. Takes
# an array/list of X and Y coordinates as inputs.
# The number of lines (count) default is 2.
def segments_fit(X, Y, count=2):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2) ** 2)

    r = minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


# Applies the piecewise linear regression method. If a row has no tongue pixels, returns that row full of 0's
def piecewise_linear(video_arr):
    segment_coords = []
    for frame in video_arr:
        if np.count_nonzero(frame) < 4:  # If there are no nonzero values
            segment_coords.append(np.full((2, 3), -1))
        else:
            y, x = frame.nonzero()
            segment_coords.append(segments_fit(x, y))
    segment_coords = np.asarray(segment_coords)
    segment_coords = np.round(segment_coords)
    return segment_coords
