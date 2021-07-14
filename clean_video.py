import numpy as np
import cv2 as cv

# https://stackoverflow.com/questions/46143800/removing-isolated-pixels-using-opencv

# Returns a cleaned background sub. This means that any pixels that
# are not adjacent to others are removed.
# Takes a black and white image (a background subtracted image) as an input


# For testing purposes
def main():
    video = np.load('./data_output/mog_bg_sub.npy')
    frame = video[447]
    cleaned = remove_isolated_pixels(frame)
    cv.imshow('frame', frame)
    cv.imshow('clean_frame', cleaned)
    while cv.waitKey(0) != 27:
        pass


# Takes a black and white video array.
# Returns video array with isolated pixels removed.
def clean_bg_sub(video_array):
    cleaned = []
    for frame in video_array:
        cleaned.append(remove_isolated_pixels(frame))
    return np.asarray(cleaned)


def remove_isolated_pixels(image):
    connectivity = 8

    output = cv.connectedComponentsWithStats(image, connectivity, cv.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label, cv.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image


# Takes a black and white video array, the coordinates (x, y) of where the meniscus
# intersects with the tongue on each frame, and the tongue maxes (x) on each frame
# to remove the meniscus and all noise to the right of the tongue max. Returns
# a black and white numpy array.
def extract_tongue_pixels(video_array, meniscus_arr, tongue_maxes):
    tongue_px = []
    video = np.copy(video_array)
    for frame_num, frame in enumerate(video):  # frame_num starts from 0
        frame[:, tongue_maxes[frame_num]:] = 0
        no_meniscus = remove_45d_to_315d(frame, meniscus_arr[frame_num])
        tongue_px.append(no_meniscus)
    # print(tongue_maxes[0])
    # print(tongue_maxes[1])
    # tongue_px[frame_num]
    tongue_px = np.asarray(tongue_px)
    return tongue_px


# Example:
# Pre, Inputted Array:
# [[255 255 255 255 255]
#  [255 255 255 255 255]
#  [255 255 255 255 255]
#  [255 255 255 255 255]
#  [255 255 255 255 255]]
# Post, where point = (2, 2)
# [[  0   0   0   0 255]
#  [  0   0   0 255 255]
#  [  0   0 255 255 255]
#  [  0   0   0 255 255]
#  [  0   0   0   0 255]]
# Takes an inputted 2d array and a point (x, y) and returns the array with
# all values from 45 degrees - 315 degrees and sets them to 0 as shown above.
def remove_45d_to_315d(input_arr, point):
    a = input_arr
    point = np.asarray(point)
    x = point[0]
    y = point[1]
    y_dim, x_dim = np.shape(a)
    a = np.triu(a, x - y)
    a = np.flipud(a)
    a = np.triu(a, -1 * (y_dim - 1 - y - x))
    a = np.flipud(a)
    return a


if __name__ == '__main__':
    main()