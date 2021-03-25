import numpy as np
from scipy import optimize
import math
import matplotlib.pyplot as plt
import cv2 as cv


# https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a

def main():
    meniscus_eq = np.asarray([-1.76432786e-02, 1.23399985e+00, 4.36404085e+01])
    points = np.load('./data_output/mog_bg_sub.npy')
    meniscus = np.load('./data_output/meniscus_points.npy')
    # points = np.load('./data_output/tongue_points.npy', allow_pickle=True)

    meniscus = meniscus[447]
    meniscus_max = np.max(meniscus)
    points = points[447]

    cv.imshow('frame', points)
    points = np.asarray(points)
    print(np.shape(points))

    points[:, :meniscus_max + 2] = 0  # setting lower bounds (+ 2 pixels to remove artifacts due to the meniscus)
    points[:, 163:] = 0  # setting upper bounds
    cv.imshow('frame2', points)
    y, x = points.nonzero()

    # x = np.arange(len(points))
    # y = points
    # x = x[40:]
    # y = points[40:]
    # print(y)

    px, py = segments_fit(x, y)
    print('x: ', px)
    print('y: ', py)

    solx, soly = intercepts(px, py, meniscus_eq)
    print(solx, soly)

    plt.plot(x, y, 'o')
    plt.plot(px, py, 'or-')
    plt.plot([solx], [soly], 'rx')
    plt.show()


def intercepts(x, y, meniscus):
    if y[0] - y[1] == 0:  # Then 1 intercept
        soly = y
        # solx =  I STOPPED WORKING HERE

    m = (x[0] - x[1]) / (y[0] - y[1])  # Reversed so the line is inverted
    intercept = -y[0] * m + x[0]  # Reversed so the line is inverted
    print(m, intercept)

    a = meniscus[0]
    b = meniscus[1] - m
    c = meniscus[2] - intercept

    # calculate the discriminant
    d = (b ** 2) - (4 * a * c)

    # find two solutions
    sol1 = (-b - math.sqrt(d)) / (2 * a)
    sol2 = (-b + math.sqrt(d)) / (2 * a)
    print(sol1)
    soly = m * sol1 + intercept

    return soly, sol1  # Reversed so the number is re-inverted


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
    main()
