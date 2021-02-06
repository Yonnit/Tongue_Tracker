import numpy as np
# from numpy.polynomial import Chebyshev as T
from scipy.signal import find_peaks


def analyse_data(tongue_xy, meniscus_shape):
    maximums = tongue_max(tongue_xy)
    minimums = meniscus_min(meniscus_shape)
    meniscus = meniscus_pos(meniscus_shape, minimums)
    return meniscus


def meniscus_pos(meniscus_shape, minimums):
    # meniscus = meniscus_shape.copy()
    # for frame_num, shape in enumerate(meniscus_shape):
    #     if frame_num == minimums:
    #         meniscus[frame_num, :] = meniscus_shape[frame_num, :]
    #     else:
    #         meniscus[frame_num, :] = meniscus[frame_num - 1, :]
    meniscus = np.zeros_like(meniscus_shape)
    for frame_num in np.arange(np.shape(meniscus_shape)[0])[:]:  # previously had loop skip first frame, example: )[1:]:
        # print(frame_num)
        if frame_num in minimums:
            meniscus[frame_num, :] = meniscus_shape[frame_num, :]
        else:
            meniscus[frame_num, :] = meniscus[frame_num - 1, :]
    return meniscus


# TODO: Make input variable for the frame rate
# To decrease the issues, might want to base it off of meniscus shape because
# that is using the no learning bg sub
# min_btwn_lick has to be variable and depend on the millisecond to frame conversion
def tongue_max(tongue_xy, min_btwn_lick=30):
    len_arr = []
    for x_max in tongue_xy:
        len_arr.append(len(x_max))

    len_arr = np.asarray(len_arr)
    maximums = find_peaks(len_arr, distance=min_btwn_lick)
    print(f'number of maximums= {len(maximums[0])}')
    # np.savetxt('./data_output/x_max.csv', len_arr, delimiter=',')
    return maximums


def meniscus_min(meniscus_shape):
    # max_x_vals = np.amax(meniscus_shape, axis=1)

    n = 2  # sets how large of a number we'll get (ie n=2 means we get the second largest number)
    nth_max = np.sort(meniscus_shape)[:, -n]

    minimums = find_peaks(-nth_max, distance=20, width=5, height=(None, 1))
    print(f'number of minimums= {len(minimums[0])}')
    # np.savetxt('./data_output/men_max.csv', nth_max, delimiter=',')
    return minimums[0]


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


# # n is the number of maxes to take the median of
# def find_median_of_maxes(arr, n):
#     idx_med = (-arr).argsort(axis=-1)[:, :n][:, 1]
#     return arr[np.arange(len(arr)), idx_med]