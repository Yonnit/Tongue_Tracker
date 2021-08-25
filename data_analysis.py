import numpy as np


def analyse_data(user_input, tongue_max_frames, segment_coords, meniscus_arr):
    num_frames = np.shape(segment_coords)[0]
    frame_num = np.arange(1, num_frames + 1)
    time_in_ms = (frame_num - 1) * user_input['fps']**-1 * 1000
    tongue_lengths = lengths(segment_coords)
    is_max = np.full(num_frames, False)
    is_max[tongue_max_frames] = True
    output_arr = np.column_stack((frame_num, time_in_ms, tongue_lengths, is_max, meniscus_arr))

    tf = output_arr[:, 3]
    tf = tf.astype(np.bool)
    just_maxes = output_arr[tf]
    return output_arr, just_maxes


def lengths(segment_coords):
    xa = segment_coords[:, 0, 0]
    xb = segment_coords[:, 0, 1]
    xc = segment_coords[:, 0, 2]
    ya = segment_coords[:, 1, 0]
    yb = segment_coords[:, 1, 1]
    yc = segment_coords[:, 1, 2]
    sum_length = segment_length(xa, xb, ya, yb) + segment_length(xb, xc, yb, yc)
    return sum_length
    # np.savetxt("./data_output/lengths.csv", sum_length, delimiter=",")


def segment_length(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# For testing
# Throws IndexError if array isn't at least 3D (More or less = problems)
def main():
    from scipy.spatial import distance
    # a = np.load('./data_output/segment_coords.npy')
    a = np.arange(20, step=2).reshape(1, 5, 2)
    b = np.diff(a, axis=1)**2

    distance.cdist()
    # np.add.reduceat(np.arange(8), [0, 4, 1, 5, 2, 6, 3, 7])[::2]
    # a = np.dsplit(a, depth)
    # for point in

    # tongue_lengths = lengths(a)
    # print(np.shape(tongue_lengths))
    print()


if __name__ == '__main__':
    main()


# import numpy as np
# from scipy.signal import find_peaks
#
# from regression import parabola_fit
# from tongue_functions import find_tongue_end, find_tongue_equations
# from find_x_maxes import find_x_maxes
#
#
# def analyse_video(cleaned_bg_sub):
#     x_maxes = find_x_maxes(cleaned_bg_sub)
#     minimums = meniscus_min(x_maxes)
#     meniscus_points = meniscus_pos(x_maxes, minimums)
#     meniscus_equations = find_meniscus_equations(meniscus_points)
#
#     tongue_maxes = find_tongue_end(cleaned_bg_sub)
#     tongue_max_frames = find_peaks(tongue_maxes, distance=30)
#     print('num tongue maxes=', len(tongue_max_frames[0]))
#
#     # select_meniscus(cleaned_bg_sub, tongue_max_frames)
#
#     # tongue_equations = find_tongue_equations(cleaned_bg_sub, meniscus_points, tongue_maxes)
#     return tongue_max_frames
#
#
# def find_meniscus_equations(meniscus_points):
#     unique_points, indices = np.unique(meniscus_points, axis=0, return_inverse=True)
#     unique_equations = np.apply_along_axis(parabola_fit, 1, unique_points)
#     return unique_equations[indices]
#
#
# def meniscus_pos(meniscus_points, minimums):
#     meniscus = np.zeros_like(meniscus_points)
#     for frame_num in np.arange(np.shape(meniscus_points)[0])[1:]:  # skips first frame (to avoid errors)
#         if frame_num in minimums:
#             meniscus[frame_num, :] = meniscus_points[frame_num, :]
#         else:
#             meniscus[frame_num, :] = meniscus[frame_num - 1, :]
#     return meniscus
#
#
# # TO DO: Make input variable for the frame rate
# # To decrease the issues, might want to base it off of meniscus shape because
# # that is using the no learning bg sub
# # min_btwn_lick has to be variable and depend on the millisecond to frame conversion
# def tongue_max(tongue_xy, min_btwn_lick=30):
#     len_arr = []
#     for x_max in tongue_xy:
#         len_arr.append(len(x_max))
#
#     len_arr = np.asarray(len_arr)
#     maximums = find_peaks(len_arr, distance=min_btwn_lick)
#     print(f'number of maximums= {len(maximums[0])}')
#     # np.savetxt('./data_output/x_max.csv', len_arr, delimiter=',')
#     return maximums
#
#
# def meniscus_min(meniscus_points):
#     # max_x_vals = np.amax(meniscus_points, axis=1)
#
#     n = 2  # sets how large of a number we'll get (ie n=2 means we get the second largest number)
#     nth_max = np.sort(meniscus_points)[:, -n]
#
#     minimums, properties = find_peaks(-nth_max, distance=20, width=5, height=(None, 1))
#     print(f'number of minimums= {len(minimums)}')
#     # np.savetxt('./data_output/men_max.csv', nth_max, delimiter=',')
#     return minimums
#
#
# def tongue_length(tongue_xy):
#     length = []
#     for frame in tongue_xy:
#         length.append(length_from_points(frame, 0, len(frame)))
#     print(length)
#     print(np.shape(length))
#     return length
