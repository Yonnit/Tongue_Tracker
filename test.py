import numpy as np
import matplotlib.pyplot as plt
# from numpy.polynomial import Chebyshev as T
from scipy.optimize import least_squares


def main():
    meniscus_points = np.load('./data_output/meniscus_df.npy')

    y = np.arange(len(meniscus_points[0]))
    x = meniscus_points[200]
    mask = x > 0  # creates mask that excludes the placeholder negative ints

    t_train = y[mask]
    y_train = x[mask]
    x0 = np.array([1.0, 1.0, 0.0])
    res_lsq = least_squares(fun, x0, args=(t_train, y_train))
    res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
                                args=(t_train, y_train))
    # res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
    #                         args=(t_train, y_train))

    t_test = np.linspace(0, 76, 77*10)  # HARDCODED IN!
    y_lsq = gen_data(t_test, *res_lsq.x)
    y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
    # y_log = gen_data(t_test, *res_log.x)

    plt.plot(t_train, y_train, 'o')
    plt.plot(t_test, y_lsq, label='linear loss')
    plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
    # plt.plot(t_test, y_log, label='cauchy loss')
    plt.xlabel("t")
    plt.ylabel("y")

    # p = T.fit(y[mask], x[mask], 2)
    # # plt.plot(y[mask], x[mask], 'o')
    # xx, yy = p.linspace()
    # plt.plot(xx, yy, lw=2, label='default')
    plt.legend()
    plt.show()
    # for y_coord, x_coord in enumerate(meniscus_points[100]):
    #     center_coordinates = x_coord, y_coord
    #     print(center_coordinates)


def gen_data(t, a, b, c,):
    # return a + b * np.exp(t * c)
    return a * t ** 2 + b * t + c


def fun(x, t, y):
    # return x[0] + x[1] * np.exp(x[2] * t) - y
    return x[0] * t ** 2 + x[1] * t + x[2] - y


if __name__ == '__main__':
    main()
