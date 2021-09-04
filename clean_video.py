import numpy as np
import cv2 as cv


# https://stackoverflow.com/questions/46143800/removing-isolated-pixels-using-opencv

# Returns a cleaned background sub. This means that any pixels that
# are not adjacent to others are removed.
# Takes a black and white image (a background subtracted image) as an input


# For testing purposes
def main():
    from multi_video_player import video_player
    start_frame = 0
    video = np.load('./data_output/B2-S20__20210805_220313/mog_bg_sub.npy')[733:]
    cleaned = clean_bg_sub(video)
    video_player(start_frame, video, cleaned)


# Takes a black and white video array.
# Returns it cleaned of isolated pixels (random noise) and distant components (usually bubbles)
def clean_bg_sub(video_array):
    print("Starting to Clean the Video")
    cleaned = []
    min_size = 100
    dist_thresh = 20
    for frame in video_array:
        # no_isolated = remove_isolated_pixels(frame)
        cleaned_frame = remove_distant_components(frame, min_size, dist_thresh)
        cleaned.append(cleaned_frame)
    params = {
        'clean_video params': {
            'minimum size(px)': min_size,
            'maximum distance threshold(px)': dist_thresh
        }
    }
    print("Done Cleaning Video")
    return np.asarray(cleaned), params


    # TODO: Change from being hardcoded to scale from cm per px
    # min_size = minimum size of particles that go un-vetted.
    # To the program, they are unquestionably either the tongue or
    # meniscus. They are kept and are what the dist_from_arr are based on.
    # dist_thresh is the maximum distance from the tongue or meniscus (as determined
    # by the min_size value) that a component is kept. Anything equal to or further
    # than the distance is deleted. The maximum distance value is from the part of each component
    # nearest one another
def remove_distant_components(image, min_size, dist_thresh):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component,
    # but most of the time we don't want that.
    sizes = stats[1:, -1];
    # if np.size(sizes) == 0:  # If there are no components
    #     return image
    centroids = centroids[1:].astype(int)
    nb_components = nb_components - 1

    # your answer image
    cleaned = np.zeros(output.shape, dtype=np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            cleaned[output == i + 1] = 255

    # Shows the distance from the nearest component that met the min_size requirement
    inverted = cv.bitwise_not(cleaned)
    dist_from_arr = cv.distanceTransform(inverted, cv.DIST_C, cv.DIST_MASK_PRECISE)

    # # Puts components that are close enough to the tongue/meniscus back into the cleaned picture
    # for i, centroid in enumerate(centroids):
    #     if dist_from_arr[centroid[1], centroid[0]] < 10:
    #         cleaned[output == i + 1] = 255

    for i in range(2, nb_components):  #
        min_distance = np.amin(dist_from_arr[output == i + 1])
        if min_distance < dist_thresh:
            cleaned[output == i + 1] = 255

    return cleaned


# Takes a binary image and removes the isolated pixels
#           Could be merged with remove distant components but I'm too lazy
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
