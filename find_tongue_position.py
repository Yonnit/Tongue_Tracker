import numpy as np


from find_contiguous_regions import contiguous_above_thresh


def find_tongue_xy_pos(bg_sub_array):
    avg_vertical = to_vertical_bands(bg_sub_array)
    tongue_x_max = find_tongue_x_max(avg_vertical)
    median_y_position = find_median_y_position(bg_sub_array)
    # print(np.shape(median_y_position))
    # print(median_y_position[350, :])

    xy_coords = combine_xy_coords(tongue_x_max, median_y_position)
    # print(np.shape(xy_coords))
    # print(xy_coords[350])
    return xy_coords


# might be removed later when mensicus shape is implemented
def combine_xy_coords(x_max, y):
    total_frames = np.shape(x_max)[0]
    frame_xy_pos = [[] for i in range(total_frames)]
    # np.empty_like(x_max, dtype=object)
    frame_num = 0
    for frame in frame_xy_pos:
        for x in range(x_max[frame_num]):
            coordinates = y[frame_num, x]
            frame.append(coordinates)
        frame_num += 1
    return frame_xy_pos


# Returns the median vertical index of the 'on' pixels from the
# background subbed array.
# A background subbed array with the values in this order:
# [frame_num, vertical_dimension, horizontal_dimension]
def find_median_y_position(bg_sub_arr):
    y_pos_arr = np.apply_along_axis(find_median_index, 1, bg_sub_arr)
    return y_pos_arr


# Returns the median index of nonzero values.
# Takes a one dimensional array of values, finds the indices
# of the non-zero values, than the median of those indices.
def find_median_index(one_dimension_array):
    indices = np.nonzero(one_dimension_array)
    median_index = np.median(indices)
    return median_index


# TODO: make save vertical array to text optional
# Returns the average brightness of a vertical slice of pixels
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def to_vertical_bands(input_array):
    # a = np.zeros(len(input_array))
    avg_vert_array = input_array.mean(axis=1)
    print(f'Frame count={avg_vert_array.shape[0]} Horizontal Resolution={avg_vert_array.shape[1]}')
    # np.savetxt('./data_output/vertical_bands_arr.csv', avg_vert_array, delimiter=',')
    return avg_vert_array


# TODO: make min_seg_length and threshold changeable by user input
def find_tongue_x_max(avg_vertical):
    frame_by_frame = np.apply_along_axis(contiguous_above_thresh, 1, avg_vertical, min_seg_length=30)
    return frame_by_frame


# def average_indices(one_dimension_array):
#     indices = np.nonzero(one_dimension_array)
#     avg_index = np.mean(indices, dtype=np.dtype(int))
#     return avg_index
