import numpy as np
from scipy.signal import find_peaks

from regression import parabola_fit
from tongue_functions import find_tongue_end, find_tongue_equations
from find_x_maxes import find_x_maxes


def analyse_video(cleaned_bg_sub):
    x_maxes = find_x_maxes(cleaned_bg_sub)
    minimums = meniscus_min(x_maxes)
    meniscus_points = meniscus_pos(x_maxes, minimums)
    meniscus_equations = find_meniscus_equations(meniscus_points)

    tongue_maxes = find_tongue_end(cleaned_bg_sub)
    tongue_max_frames = find_peaks(tongue_maxes, distance=30)
    print('num tongue maxes=', len(tongue_max_frames[0]))

    # select_meniscus(cleaned_bg_sub, tongue_max_frames)

    # tongue_equations = find_tongue_equations(cleaned_bg_sub, meniscus_points, tongue_maxes)
    return tongue_max_frames


def find_meniscus_equations(meniscus_points):
    unique_points, indices = np.unique(meniscus_points, axis=0, return_inverse=True)
    unique_equations = np.apply_along_axis(parabola_fit, 1, unique_points)
    return unique_equations[indices]


def meniscus_pos(meniscus_points, minimums):
    meniscus = np.zeros_like(meniscus_points)
    for frame_num in np.arange(np.shape(meniscus_points)[0])[1:]:  # skips first frame (to avoid errors)
        if frame_num in minimums:
            meniscus[frame_num, :] = meniscus_points[frame_num, :]
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


def meniscus_min(meniscus_points):
    # max_x_vals = np.amax(meniscus_points, axis=1)

    n = 2  # sets how large of a number we'll get (ie n=2 means we get the second largest number)
    nth_max = np.sort(meniscus_points)[:, -n]

    minimums, properties = find_peaks(-nth_max, distance=20, width=5, height=(None, 1))
    print(f'number of minimums= {len(minimums)}')
    # np.savetxt('./data_output/men_max.csv', nth_max, delimiter=',')
    return minimums


def tongue_length(tongue_xy):
    length = []
    for frame in tongue_xy:
        length.append(length_from_points(frame, 0, len(frame)))
    print(length)
    print(np.shape(length))
    return length
