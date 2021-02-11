import numpy as np

from find_contiguous_regions import contiguous_above_thresh


def find_x_maxes(bg_sub_array):
    pos = np.apply_along_axis(contiguous_above_thresh, 2, bg_sub_array, min_seg_length=5, is_reversed=True)
    # print(np.shape(pos))
    # print(pos[350, :])
    return pos
