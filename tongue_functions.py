import numpy as np

import cv2 as cv
import matplotlib.pyplot as plt

from find_contiguous_regions import contiguous_above_thresh
from regression import segments_fit


# Returns the x value of the end of the tongue.
# Takes a background sub as an input (mog2 is the best for this)
def find_tongue_end(bg_sub_array):
    num_vert = count_pixels_vert(bg_sub_array)
    tongue_x_max = find_tongue_x_max(num_vert)

    return tongue_x_max


# TODO: make save vertical array to text optional
# Returns the number of bright pixels from each vertical slice of pixels starting from x=0
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def count_pixels_vert(input_array):
    # a = np.zeros(len(input_array))
    # avg_vert_array = input_array.mean(axis=1)
    count_vert_nonzero = np.count_nonzero(input_array, axis=1)
    print(f'Frame count={count_vert_nonzero.shape[0]} Horizontal Resolution={count_vert_nonzero.shape[1]}')
    # np.savetxt('./data_output/vertical_bands_arr.csv', avg_vert_array, delimiter=',')
    return count_vert_nonzero


# TODO: make min_seg_length and threshold changeable by user input
def find_tongue_x_max(num_vertical_pixels):
    frame_by_frame = np.apply_along_axis(contiguous_above_thresh, 1, num_vertical_pixels, min_seg_length=30)
    return frame_by_frame


def find_tongue_equations(tongue_points, meniscus, tongue_maxes):
    for i in np.arange(len(tongue_points)):
        key = cv.waitKey(2)  # waits 8ms between frames
        if key == 27:  # if ESC is pressed, exit loop
            break
        print(i)
        a(tongue_points, meniscus, tongue_maxes, i)


def a(tongue_points, meniscus, tongue_maxes, i):
    meniscus = meniscus[i]
    tongue_max = tongue_maxes[i]
    tongue_points = tongue_points[i]

    meniscus_max = np.max(meniscus)

    cv.imshow('frame', tongue_points)
    tongue_points = np.asarray(tongue_points)

    tongue_points[:, :meniscus_max + 2] = 0  # setting lower bounds (+ 2 pixels to remove artifacts due to the meniscus)
    tongue_points[:, tongue_max:] = 0  # setting upper bounds
    cv.imshow('frame2', tongue_points)
    y, x = tongue_points.nonzero()

    if x.size == 0:
        return -1  # Returns this if there are no pixels. (stops errors)
    px, py = segments_fit(x, y)
    print('x: ', px)
    print('y: ', py)

    return px, py
    # plt.plot(x, y, 'o')
    # plt.plot(px, py, 'or-')
    # plt.show()



# For testing purposes
if __name__ == '__main__':
    arr = np.load('./data_output/mog_bg_sub.npy')
    find_tongue_end(arr)

