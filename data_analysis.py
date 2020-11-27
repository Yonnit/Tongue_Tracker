import numpy as np
from math import hypot


def tongue_length(tongue_xy):
    length = []
    for frame in tongue_xy:
        length.append(length_from_points(frame, 0, len(frame)))
    print(length)
    print(np.shape(length))
    return length


def length_from_points(point_arr, start, stop):
    x_range = range(start, stop - 1)
    total_length = 0
    for x in x_range:
        # if np.isnan(point_arr[x]) | np.isnan(point_arr[x + 1]):
        #     dist_adjacent =
        point1 = np.array(x, point_arr[x])
        point2 = np.array(x + 1, point_arr[x + 1])
        dist_adjacent = np.linalg.norm(point1 - point2)
        total_length += dist_adjacent
    return total_length


def is_licking()

# def polynomial_regression(tongue_xy):
#     line_eq = []
#     for frame in tongue_xy:
#         if len(frame) > 0:
#             coefficients = np.polyfit(range(len(frame)), frame, 2)
#             function = np.poly1d(coefficients)
#             line_eq.append(function)
#         else:
#             line_eq.append(0)
#     print(np.shape(line_eq))
#     print(line_eq[350])
#     print(line_eq[351])
#     print(line_eq[352])
#     return line_eq
