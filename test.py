import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import cv2 as cv


# https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a

def main(tongue_points, meniscus, i):
    # tongue_points = np.load('./data_output/tongue_points.npy', allow_pickle=True)
    print(np.arange(np.shape(tongue_points)[0]))
    meniscus = meniscus[i]
    meniscus_max = np.max(meniscus)
    tongue_points = tongue_points[i]

    cv.imshow('frame', tongue_points)
    tongue_points = np.asarray(tongue_points)
    print(np.shape(tongue_points))

    tongue_points[:, :meniscus_max + 2] = 0  # setting lower bounds (+ 2 pixels to remove artifacts due to the meniscus)
    tongue_points[:, -1:] = 0  # setting upper bounds
    cv.imshow('frame2', tongue_points)
    y, x = tongue_points.nonzero()

    # x = np.arange(len(tongue_points))
    # y = tongue_points
    # x = x[40:]
    # y = tongue_points[40:]
    # print(y)

    px, py = segments_fit(x, y)
    print('x: ', px)
    print('y: ', py)

    plt.plot(x, y, 'o')
    plt.plot(px, py, 'or-')
    plt.show()


def intercepts(x1, x2, y1, y2):
    m = (y1 - y2) / (x1 - x2)
    b = -x1 * m + y1

    # x = ((-B + m + - Sqrt[4 A b + B ^ 2 - 4 A C - 2 B m + m ^ 2]) / (2 A))


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

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


if __name__ == '__main__':
    tongue_points = np.load('./data_output/mog_bg_sub.npy')
    meniscus = np.load('./data_output/meniscus_points.npy')
    for i in np.arange(np.size(tongue_points[0])):
        main(tongue_points, meniscus, i)

