import numpy as np
import cv2 as cv
import argparse
import sys

from select_tube import get_tube


def main():
    input_path = file_and_path()
    zoomed_video_arr = get_tube(input_path)
    bg_sub_array = background_subtract(zoomed_video_arr)

    avg_vertical = to_vertical_bands(bg_sub_array)
    a = find_tongue_max(avg_vertical)
    line_vid_arr = show_position(a, zoomed_video_arr)
    save_arr_to_video(line_vid_arr, "./data_output/estimated_position.avi", 20)


# Takes the estimated x position and a cropped video array
# Displays, then Returns a video array with the estimated x-position line
# included in it.
def show_position(estimated_position, video_to_compare_arr):
    (frame_height, frame_width, rgb_intensities) = video_to_compare_arr[0].shape
    line_video_arr = []
    frame_num = 0
    for frame in video_to_compare_arr:
        # print(frame_num)
        start_point = estimated_position[frame_num], 0
        end_point = estimated_position[frame_num], frame_height
        color = (0, 255, 0)
        thickness = 1
        frame = cv.line(frame, start_point, end_point, color, thickness)
        cv.imshow('frame', frame)
        line_video_arr.append(frame)
        key = cv.waitKey(8)  # waits 8ms between frames
        if key == 27:  # if ESC is pressed, exit loop
            break
        frame_num += 1
    cv.destroyAllWindows()
    line_video_arr = np.asarray(line_video_arr)
    return line_video_arr


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


def find_tongue_max(avg_vertical):
    a = np.apply_along_axis(contiguous_above_thresh, 1, avg_vertical)
    return a


# Takes black and white vector as input and returns the first frame that
# is above a certain intensity for a certain number of pixels as defined
# by segment
def contiguous_above_thresh(row):
    threshold = 5  # TODO: Make this and the segment var a parameter passed from input args
    condition = row > threshold  # Creates array of boolean values (True = above threshold)
    # print('Row Break')  # AKA new frame
    for start, stop in contiguous_regions(condition):  # For every
        # print('In For Loop')
        segment = row[start:stop]
        if len(segment) > 30:  # If the above threshold pixels extend across length greater than 20 pixels return
            return start
    return -1  # There were no segments with greater than 20 pixels above threshold


# TODO: make save vertical array to text optional
# Returns the average brightness of a vertical slice of pixels
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def to_vertical_bands(input_array):
    # a = np.zeros(len(input_array))
    avg_vert_array = input_array.mean(axis=1)
    print(f'Frame count, Horizontal Resolution: {avg_vert_array.shape}')
    # np.savetxt('./data_output/foo.csv', avg_vert_array, delimiter=',')
    return avg_vert_array


# TODO: allow user to set --algo to switch between MOG2 and KNN
# Applies opencv's background subtract method to the inputted video
# and returns that video as an array of frames that contain frame data.
# The input must be an array of frames starting from 0
def background_subtract(input_video_arr):
    # total_frame_count = len(input_video_arr)
    # print(f'Total number of frames: {total_frame_count}')
    print('Starting Background Subtract')
    back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30, detectShadows=False)
    bg_subbed_vid_arr = []
    for frame in input_video_arr:
        foreground_mask = back_sub.apply(frame)
        bg_subbed_vid_arr.append(foreground_mask)

    bg_subbed_vid_arr = np.asarray(bg_subbed_vid_arr)
    print('Done with Background Subtract')
    return bg_subbed_vid_arr


def save_arr_to_video(arr_video_input, output_file_path, fps):
    print("Saving to video")
    (frame_height, frame_width, rgb_intensities) = arr_video_input[0].shape
    # Define the codec, create VideoWriter object.
    output = cv.VideoWriter(output_file_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            fps, (frame_width, frame_height), True)
    for frame in arr_video_input:
        output.write(frame)
    output.release()
    cv.destroyAllWindows()
    print(f"Saved to {output_file_path}")


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

########################################################################################
# Maybe stuff:

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image file")
# ap.add_argument("-c", "--coords",
# 	help = "comma seperated list of source points")
# args = vars(ap.parse_args())

##########################################################################################
# Junkyard

# def first_above_value(row):
#     threshold = 20
#     return np.argmax(row > threshold)

# def view_video():
#     video_path = file_and_path()
#     cap = cv.VideoCapture(video_path)
#
#     # Checks if file opened
#     if not cap.isOpened():
#         print("Error opening video file")
#         sys.exit(-1)
#
#     cv.namedWindow('Frame', cv.WINDOW_AUTOSIZE)
#
#     # Read until end of video
#     while cap.isOpened():
#         (exists_frame, frame) = cap.read()
#
#         # Displays resulting frame
#         if exists_frame:
#             cv.imshow('Frame', frame)
#
#             # Press Q to exit
#             if cv.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     cv.destroyAllWindows()
#     print("Success...?")

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
