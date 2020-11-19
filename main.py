import numpy as np
import cv2 as cv
# import argparse
# import sys
# import scipy.stats as stats

from select_tube import get_tube


def main():
    # input_path = file_and_path()
    # zoomed_video_arr = get_tube(input_path)
    # bg_sub_array = background_subtract(zoomed_video_arr)
    #
    # np.save('./data_output/cropped_video', zoomed_video_arr)
    # np.save('./data_output/bg_sub', bg_sub_array)

    zoomed_video_arr = np.load('./data_output/cropped_video.npy')
    bg_sub_array = np.load('./data_output/bg_sub.npy.')
    # print(np.shape(bg_sub_array))

    avg_vertical = to_vertical_bands(bg_sub_array)

    tongue_x_pos = find_tongue_x_max(avg_vertical)
    tongue_y_pos = find_tongue_y_pos(bg_sub_array)
    print(np.shape(tongue_y_pos))

    # np.save('./data_output/mode_vertical', mode_vertical)
    # mode_vertical = np.load('./data_output/mode_vertical.npy')
    # np.savetxt('./data_output/meniscus_x_position.csv', mode_vertical[0, :, :], delimiter=',')
    # print(np.shape(mode_vertical))

    # np.savetxt('./data_output/meniscus_x_position.csv', meniscus_x_pos, delimiter=',')
    np.savetxt('./data_output/tongue_x_position.csv', tongue_x_pos, delimiter=',')

    dot_vid_arr = show_tongue_loc(tongue_x_pos, tongue_y_pos, zoomed_video_arr)
    save_arr_to_video(dot_vid_arr, "tongue_position", 20, True)

    # line_vid_arr = show_position(tongue_x_pos, bg_sub_array, False)  # , meniscus_x_pos (add to end later)
    # save_arr_to_video(line_vid_arr, "estimated_position", 20, False)


def show_tongue_loc(tongue_x_pos, tongue_y_pos, video_to_compare_arr):
    color = (0, 255, 0)
    (frame_height, frame_width, rgb_intensities) = video_to_compare_arr[0].shape

    line_video_arr = []
    frame_num = 0
    for frame in video_to_compare_arr:
        print(frame_num)

        thickness = 1
        radius = 1

        x_val = np.arange(tongue_x_pos[frame_num])  # might have to do tongue_x_pos[frame_num] + 1 if values are cut off
        for x in x_val:
            center_coordinates = x, tongue_y_pos[frame_num, x]
            frame = cv.circle(frame, center_coordinates, radius, color, thickness)

        cv.imshow('frame', frame)
        line_video_arr.append(frame)
        key = cv.waitKey(50)  # waits 8ms between frames
        if key == 27:  # if ESC is pressed, exit loop
            break
        frame_num += 1
    cv.destroyAllWindows()
    line_video_arr = np.asarray(line_video_arr)
    return line_video_arr


# Takes the estimated x position and a cropped video array
# Displays, then Returns a video array with the estimated x-position line
# included in it. Can include additional inputted positions in the
# *args.
def show_position(estimated_position, video_to_compare_arr, is_color, *args):
    if is_color:
        color = (0, 255, 0)
        (frame_height, frame_width, rgb_intensities) = video_to_compare_arr[0].shape
    else:
        color = (255, 255, 255)
        (frame_height, frame_width) = video_to_compare_arr[0].shape

    line_video_arr = []
    frame_num = 0
    for frame in video_to_compare_arr:
        # print(frame_num)

        start_point = estimated_position[frame_num], 0
        end_point = estimated_position[frame_num], frame_height
        thickness = 1
        frame = cv.line(frame, start_point, end_point, color, thickness)

        for arg in args:
            start_point = arg[frame_num], 0
            end_point = arg[frame_num], frame_height
            frame = cv.line(frame, start_point, end_point, color, thickness)

        cv.imshow('frame', frame)
        line_video_arr.append(frame)
        key = cv.waitKey(50)  # waits 8ms between frames
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


def find_tongue_x_max(avg_vertical):
    frame_by_frame = np.apply_along_axis(contiguous_above_thresh, 1, avg_vertical, threshold=1, min_seg_length=30)
    return frame_by_frame


def find_tongue_y_pos(bg_sub_arr):
    y_pos_arr = np.apply_along_axis(average_indices, 1, bg_sub_arr)
    return y_pos_arr


def average_indices(one_dimension_array):
    indices = np.nonzero(one_dimension_array)
    avg_index = np.mean(indices, dtype=np.dtype(int))
    return avg_index


# Takes black and white vector as input and returns the first frame that
# is above a certain intensity for a certain number of pixels as defined
# by segment
def contiguous_above_thresh(row, threshold, min_seg_length):
    condition = row > threshold  # Creates array of boolean values (True = above threshold)
    # print('Row Break')  # AKA new frame
    for start, stop in contiguous_regions(condition):  # For every
        # print('In For Loop')
        segment = row[start:stop]
        # If the above threshold pixels extend across length greater than
        # min_seg_length pixels return
        if len(segment) > min_seg_length:
            return stop  # If the x axis is flipped, should return stop.
    return -1  # There were no segments longer than the minimum length with greater intensity than threshold


# TODO: make save vertical array to text optional
# Returns the average brightness of a vertical slice of pixels
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def to_vertical_bands(input_array):
    # a = np.zeros(len(input_array))
    avg_vert_array = input_array.mean(axis=1)
    print(f'Frame count, Horizontal Resolution: {avg_vert_array.shape}')
    # np.savetxt('./data_output/vertical_bands_arr.csv', avg_vert_array, delimiter=',')
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
        foreground_mask = back_sub.apply(frame)  # try: back_sub.apply(frame, learningRate=0)
        bg_subbed_vid_arr.append(foreground_mask)

    bg_subbed_vid_arr = np.asarray(bg_subbed_vid_arr)
    print('Done with Background Subtract')
    return bg_subbed_vid_arr


# Takes a video array, the desired output file name, and the desired fps of the
# outputted file. Saves an .avi video to the data_output folder with the desired name.
def save_arr_to_video(arr_video_input, output_name, fps, is_color):
    output_path_and_name = f"./data_output/{output_name}.avi"
    print("Saving to video")

    # Define the codec, create VideoWriter object.
    if is_color:
        (frame_height, frame_width, rgb_intensities) = arr_video_input[0].shape
        output = cv.VideoWriter(output_path_and_name, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                fps, (frame_width, frame_height), True)
    else:
        (frame_height, frame_width) = arr_video_input[0].shape
        output = cv.VideoWriter(output_path_and_name, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                fps, (frame_width, frame_height), False)
    for frame in arr_video_input:
        output.write(frame)
    output.release()
    cv.destroyAllWindows()
    print(f"Saved to {output_path_and_name}")


# Prompts user to input file name
# Returns path and name in directory
# TODO: configure argparse so user selects file location, and other params
def file_and_path():  # Will likely use argparse here
    input_file_name = "vid_from_imgs.avi"
    path = "./video_input/"
    return path + input_file_name


# makes main() a main function similar to Java, C, C++
if __name__ == '__main__':
    main()

########################################################################################
# Maybe stuff:

# mode_vertical = mode_vert_bands(bg_sub_array)
# meniscus_x_pos = find_meniscus_pos(mode_vertical[0, :, :])
# def mode_vert_bands(input_array):
#     mode_vert_array = stats.mode(input_array, axis=1)
#     mode_vert_array = np.asarray(mode_vert_array)
#     mode_vert_array = np.squeeze(mode_vert_array)
#     return mode_vert_array


# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image file")
# ap.add_argument("-c", "--coords",
# 	help = "comma seperated list of source points")
# args = vars(ap.parse_args())

##########################################################################################
# Junkyard


# def find_meniscus_pos(mode_vertical):
#     x_loc_mode = np.argmax(mode_vertical, axis=1)
#     print(x_loc_mode.size)
#     min_x = np.amax(x_loc_mode)
#     min_x_arr = []
#     for x in x_loc_mode:
#         if (x != 0) & (x < min_x):
#             min_x = x
#         min_x_arr.append(min_x)
#     return min_x_arr
#
#
# def median_vert_bands(input_array):
#     return np.median(input_array, axis=1)

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


# The only way to get clean meniscus data as far as I can tell is by scanning from right to left
# and using the argmax approach. It only saves the max value if its further from the opening than
# the previous max value, and only if the change in x isn't too great (for random noise and jumps)

# # A robust but unconventional way would be to use the find max method, then
# # take the array and make it only allow the longest x values and saves that
# def find_meniscus(avg_vertical):
#     # frame_by_frame = np.apply_along_axis(contiguous_above_thresh, 1, avg_vertical, threshold=100, min_seg_length=10)
#     # return frame_by_frame
#     threshold = 80
#     max_arr = np.argmax(avg_vertical, axis=1)
#     max_arr[max_arr < threshold] = -1
#     return max_arr
