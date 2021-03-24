import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import cv2 as cv


# https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a

def main():
    points = np.load('./data_output/mog_bg_sub.npy')
    # points = np.load('./data_output/tongue_points.npy', allow_pickle=True)
    points = points[447]
    cv.imshow('frame', points)
    points = np.asarray(points)
    print(np.shape(points))

    points[:, :70] = 0  # setting lower bounds
    points[:, 166:] = 0  # setting upper bounds
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

    plt.plot(x, y, 'o')
    plt.plot(px, py, 'or-')
    plt.show()


def intercepts(x1, x2, y1, y2):
    m = (y1 - y2) / (x1 - x2)
    b = -x1 * slope + y1



    # x = ((-B + m + - Sqrt[4 A b + B ^ 2 - 4 A C - 2 B m + m ^ 2]) / (2 A))



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
