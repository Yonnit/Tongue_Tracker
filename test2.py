import numpy as np
import cv2 as cv
import sys


def main():
    a = np.load('./data_output/num_vert.npy')
    # a = a[377]
    a = np.zeros(300)
    a[0:50] = 1
    a[100:140] = 1
    ans = contiguous_above_thresh(a, 30)
    print(ans)


# Takes black and white vector as input and returns the first frame that
# is above a certain intensity for a certain number of pixels as defined
# by segment
def contiguous_above_thresh(row, min_seg_length, threshold=1):
    condition = row >= threshold  # Creates array of boolean values (True = above threshold)
    for start, stop in reversed(contiguous_regions(condition)):
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


# def main():
#     a = np.full((5, 5), 255, np.uint8)
#     print(a)
#     point = np.array((2, 2))
#     x = point[0]
#     y = point[1]
#     y_dim, x_dim = np.shape(a)
#     #
#     # print(point - dimensions)
#     # k1, k2 = point-dimensions
#
#     a = np.triu(a, x - y)
#     print('cut!')
#     print(a)
#     a = np.flipud(a)
#     print('flip!')
#     print(a)
#     a = np.triu(a, -1 * (y_dim - 1 - y - x))
#     print('cut2!')
#     print(a)
#     a = np.flipud(a)
#     print('flip2!')
#     print(a)
#     # while True:
#     #     cv.imshow('frame', a)
#     #     key = cv.waitKey(1) & 0xFF
#     #     if key == 27 or key == ord("q"):  # If the user presses Q or ESC
#     #         cv.destroyAllWindows()
#     #         sys.exit(-1)
#     # cv.destroyAllWindows()



if __name__ == '__main__':
    main()
