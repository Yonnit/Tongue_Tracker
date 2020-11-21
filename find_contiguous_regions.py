import numpy as np


def largest_contiguous_region(row, threshold):
    condition = row > threshold
    a = []
    for start, stop in contiguous_regions(condition):
        segment = row[start:stop]
        a.append(len(segment), start, stop)
    a = np.asarray(a)
    print('WORK IN PROGRESS')
    # Have to find shape of asarray array so I can find argmax of that dimension, then return the start stop
    # as a tuple
    # np.argmax(a,)


# Takes black and white vector as input and returns the first frame that
# is above a certain intensity for a certain number of pixels as defined
# by segment
def contiguous_above_thresh(row, min_seg_length, threshold=1, is_reversed=False):
    condition = row > threshold  # Creates array of boolean values (True = above threshold)
    # print('Row Break')  # AKA new frame
    if is_reversed:
        for start, stop in reversed(contiguous_regions(condition)):  # For every
            # print('In For Loop')
            segment = row[start:stop]
            # If the above threshold pixels extend across length greater than
            # min_seg_length pixels return
            if len(segment) > min_seg_length:
                return stop
    else:
        # Trying to reverse it elsewhere DOES NOT WORK! It's stupid but this is the best I could do
        for start, stop in contiguous_regions(condition):
            segment = row[start:stop]
            if len(segment) > min_seg_length:
                return stop
    return -1  # There were no segments longer than the minimum length with greater intensity than threshold





# Finds contiguous True regions of the boolean array "condition". Returns
# a 2D array where the first column is the start index of the region and the
# second column is the end index.
def contiguous_regions(condition):
    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx
