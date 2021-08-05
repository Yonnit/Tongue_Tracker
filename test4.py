import numpy as np
import cv2 as cv



# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
def main():

    video = np.load('./data_output/mog_bg_sub.npy')
    frame = video[447]


    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(frame, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component,
    # but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    min_size = np.mean(sizes)

    # your answer image
    cleaned = np.zeros(output.shape, dtype=np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            cleaned[output == i + 1] = 255

    inverted = cv.bitwise_not(cleaned)
    inv_clean = cv.distanceTransform(inverted, cv.DIST_C, cv.DIST_MASK_PRECISE)

    cv.imshow('inverted', inverted)
    cv.imshow('frame', frame)
    cv.imshow('clean_frame', cleaned)

    while cv.waitKey(0) != 27:
        pass

if __name__ == '__main__':
    main()
