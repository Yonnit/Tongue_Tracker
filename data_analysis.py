import numpy as np
from math import hypot


def tongue_length(tongue_xy):
    polynomial_approximation = polynomial_regression(tongue_xy)
    length_approx(tongue_xy, polynomial_approximation)


# """Compute the arc length of the curve defined by y = x**2 + 2*x for
# -2 <= x <= 0.5 without using calculus.
# """
def arclength(f, start, stop, tol=1e-6):
    # """Compute the arc length of function f(x) for a <= x <= b. Stop
    # when two consecutive approximations are closer than the value of
    # tol.
    # """
    nsteps = 1  # number of steps to compute
    oldlength = 1.0e20
    length = 1.0e10
    while abs(oldlength - length) >= tol:
        nsteps *= 2
        fx1 = f(start)
        xdel = (stop - start) / nsteps  # space between x-values
        oldlength = length
        length = 0
        for i in range(1, nsteps + 1):
            fx0 = fx1  # previous function value
            fx1 = f(start + i * (stop - start) / nsteps)  # new function value
            length += hypot(xdel, fx1 - fx0)  # length of small line segment
    return length


def f(x, coef):
    return coef[0]*x**2 + coef[1]*x + coef[2]


# def tongue_submersion(tongue_xy, meniscus_yx):
#     for frame_num,
#     is_licking = tongue_xy[frame_num]
#     if is_licking:
#

def polynomial_regression(tongue_xy):
    line_eq = []
    for frame in tongue_xy:
        if len(frame) > 0:
            coefficients = np.polyfit(range(len(frame)), frame, 2)
            function = np.poly1d(coefficients)
            line_eq.append(function)
        else:
            line_eq.append(0)
    print(np.shape(line_eq))
    print(line_eq[350])
    return line_eq
