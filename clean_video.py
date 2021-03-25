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


if __name__ == '__main__':
    main()