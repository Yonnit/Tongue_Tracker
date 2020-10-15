import numpy as np
import cv2 as cv
import argparse
import sys

from select_tube import get_tube
# TODO: Make saving video to output file function that takes array

def main():
    input_path = file_and_path()
    zoomed_video_arr = get_tube(input_path)
    bg_sub_array = background_subtract(zoomed_video_arr)

    avg_vertical = to_vertical_bands(bg_sub_array)
    a = find_tongue_max(avg_vertical)
    show_position(a, bg_sub_array)
# TODO: REFACTOR YOUR SHITTY CODE!!!!! BREAK STUFF INTO ITS PARTS! SEPARATE INTO DIFF FILES!!


def show_position(estimated_position, background):
    # avg_vertical = np.genfromtxt('./data_output/foo.csv', delimiter=',')
    # a = np.apply_along_axis(first_above_value, 1, avg_vertical)
    # np.savetxt('./data_output/test.csv', a, delimiter=',')
    frame = 0
    print(np.size(background))
    while frame < np.size(background, 0):
        print(frame)
        image = background[frame]

        startpoint = estimated_position[frame], 0
        endpoint = estimated_position[frame], 100
        color = (225, 255, 225)
        thickness = 2
        image = cv.line(image, startpoint, endpoint, color, thickness)
        cv.imshow('frame', image)
        frame += 1
        key = cv.waitKey(1000)  # pauses for 3 seconds before fetching next image
        if key == 27:  # if ESC is pressed, exit loop
            cv.destroyAllWindows()
            break


def first_above_value(row):
    threshold = 20  # TODO: Make this a parameter passed from input args
    return np.argmax(row > threshold)


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
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def find_tongue_max(avg_vertical):
    a = np.apply_along_axis(contiguous_above_thresh, 1, avg_vertical)
    return a


def contiguous_above_thresh(row):
    threshold = 5
    condition = row > threshold  # Creates array of boolean values (True = above threshold)
    print('Row Break')  # AKA new frame
    for start, stop in contiguous_regions(condition):
        print('In For Loop')
        segment = row[start:stop]
        if len(segment) > 20:  # If the above threshold pixels extend across length greater than 20 pixels return
            return start
    return -1  # There were no segments with greater than 20 pixels above threshold


# Returns the average brightness of a vertical slice of pixels
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def to_vertical_bands(input_array):
    # a = np.zeros(len(input_array))
    avg_vert_array = input_array.mean(axis=1)
    print(f'Frame count, Horizontal Resolution: {avg_vert_array.shape}')
    np.savetxt('./data_output/foo.csv', avg_vert_array, delimiter=',')
    return avg_vert_array


# TODO: allow user to set --algo to switch between MOG2 and KNN
# Applies opencv's background subtract method to the inputted video
# and returns that video as an array of
# The input must be an array of frames starting from 0
def background_subtract(input_video_arr):
    total_frame_count = len(input_video_arr)
    print(f'Total number of frames: {total_frame_count}')
    print('Starting background Subtract')
    back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30, detectShadows=False)
    bg_subbed_vid_arr = []
    for frame in input_video_arr:
        foreground_mask = back_sub.apply(frame)
        bg_subbed_vid_arr.append(foreground_mask)

    bg_subbed_vid_arr = np.asarray(bg_subbed_vid_arr)
    return bg_subbed_vid_arr


def view_video():
    video_path = file_and_path()
    cap = cv.VideoCapture(video_path)

    # Checks if file opened
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(-1)

    cv.namedWindow('Frame', cv.WINDOW_AUTOSIZE)

    # Read until end of video
    while cap.isOpened():
        (exists_frame, frame) = cap.read()

        # Displays resulting frame
        if exists_frame:
            cv.imshow('Frame', frame)

            # Press Q to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()
    print("Success...?")


# Prompts user to input file name
# Returns path and name in directory
# TODO: configure argparse so user selects file location, and other params
def file_and_path():  # Will likely use argparse here
    input_file_name = "Export_20171211_015532_PM.avi"
    path = "./video_input/"
    return path + input_file_name


# makes main() a main function similar to Java, C, C++
if __name__ == '__main__':
    main()

# Maybe stuff:

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image file")
# ap.add_argument("-c", "--coords",
# 	help = "comma seperated list of source points")
# args = vars(ap.parse_args())


# # Simple background subtraction
# # Saves video to output file
# # Returns a 3d numpy matrix containing the video information
# # TOD: allow user to set --algo to switch between MOG2 and KNN
# # TOD: Make saving video to output file optional
# # TOD: When next testing background sub method, remember that you changed your mind
# #   so file_and_path should be inputted as a parameter instead of directly into file
# # TOD: Remember that bg_sub should no longer display frame, separate out into diff
# #   methods like watch_video, save_video and such
# def old_background_subtract(transform_data):
#     back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30, detectShadows=False)
#     cap = cv.VideoCapture(file_and_path())
#     if not cap.isOpened():
#         print('Error opening video file')
#         sys.exit(-1)
#     # Obtains default resolution of frame & converts from float to int
#     # frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     # frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#     total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
#     # print(f'Frame resolution: Width={frame_width} Height={frame_height}')
#     print(f'Total number of frames: {total_frame_count}')
#     # # Define the codec, create VideoWriter object.
#     # output = cv.VideoWriter('./data_output/Background_Sub.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
#     #                         3, (frame_width, frame_height), 0)
#     video_array = []  # array of frames, index = frame # starting from 0
#     while True:
#         (exists_frame, frame) = cap.read()
#
#         if not exists_frame:
#             break
#         cropped_frame = apply_four_point_transform(frame, transform_data)
#         fg_mask = back_sub.apply(cropped_frame)
#         # Puts the frame count on the original video
#         # frame_number starts from 1
#         # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
#         # cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#         #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#
#         video_array.append(fg_mask)
#         # output.write(fg_mask)
#         # cv.imshow('Frame', frame)
#         # cv.imshow('FG Mask', fg_mask)
#         # if cv.waitKey(1) & 0xFF == ord('q'):
#         #     break
#     cap.release()
#     # output.release()
#     cv.destroyAllWindows()
#
#     video_array = np.asarray(video_array)
#     return video_array