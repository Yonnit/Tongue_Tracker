import numpy as np


from find_contiguous_regions import contiguous_regions



def find_tongue_y_pos(bg_sub_arr):
    y_pos_arr = np.apply_along_axis(average_indices, 1, bg_sub_arr)
    return y_pos_arr


def average_indices(one_dimension_array):
    indices = np.nonzero(one_dimension_array)
    avg_index = np.mean(indices, dtype=np.dtype(int))
    return avg_index


# Takes black and white vector as input and returns the first frame that
# is above a certain intensity for a certain number of pixels as defined
# by segment
def contiguous_above_thresh(row, threshold, min_seg_length):
    condition = row > threshold  # Creates array of boolean values (True = above threshold)
    # print('Row Break')  # AKA new frame
    for start, stop in contiguous_regions(condition):  # For every
        # print('In For Loop')
        segment = row[start:stop]
        # If the above threshold pixels extend across length greater than
        # min_seg_length pixels return
        if len(segment) > min_seg_length:
            return stop  # If the x axis is flipped, should return stop.
    return -1  # There were no segments longer than the minimum length with greater intensity than threshold


# def tongue_xy_pos(bg_sub_array):
#     avg_vertical = to_vertical_bands(bg_sub_array)
#
#     tongue_x_max = find_tongue_x_max(avg_vertical)
#     tongue_y_pos = find_tongue_y_pos(bg_sub_array)
#     print(np.shape(tongue_y_pos))
#     print(tongue_y_pos[350, :])
#
#     xy_coords = combine_xy_coords(tongue_x_max, tongue_y_pos)
#     return xy_coords


# # TODO: make save vertical array to text optional
# # Returns the average brightness of a vertical slice of pixels
# # Index represents frame starting from 0
# # Within the frame element [y,x] is the pixel location
# # ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
# def to_vertical_bands(input_array):
#     # a = np.zeros(len(input_array))
#     avg_vert_array = input_array.mean(axis=1)
#     print(f'Frame count, Horizontal Resolution: {avg_vert_array.shape}')
#     # np.savetxt('./data_output/vertical_bands_arr.csv', avg_vert_array, delimiter=',')
#     return avg_vert_array

# # Returns array of frames and x and y coordinates of the tongue. If the
# def combine_xy_coords(x_max, y):
#     frame_yx_pos = []
#     frame_num = 0
#     for frame in x_max:
#         x_val = np.arange(tongue_x_pos[frame_num])  # might have to do tongue_x_pos[frame_num] + 1 if values are off
#         for x in x_val:
#             center_coordinates = x, tongue_y_pos[frame_num, x]
#
#     x_val = np.arange(tongue_x_pos[frame_num])  # might have to do tongue_x_pos[frame_num] + 1 if values are cut off
#     for x in x_val:
#         center_coordinates = x, tongue_y_pos[frame_num, x]
#         frame = cv.circle(frame, center_coordinates, radius, color, thickness)

# # TODO: make min_seg_length and threshold changeable by user input
# def find_tongue_x_max(avg_vertical):
#     frame_by_frame = np.apply_along_axis(contiguous_above_thresh, 1, avg_vertical, threshold=1, min_seg_length=30)
#     return frame_by_frame
