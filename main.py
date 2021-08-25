import numpy as np
import cv2 as cv
from scipy.signal import find_peaks
import argparse
import os.path
import sys

from select_tube import get_tube
from clean_video import clean_bg_sub, extract_tongue_pixels
from tongue_functions import find_tongue_end
from select_meniscus import get_meniscus
from data_analysis import analyse_data
from regression import piecewise_linear
from output_functions import save_results, mkdir_from_input


def main():
    # zoomed_video_arr = np.load('./data_output/B1-S60_FULL__20210816_202626/zoomed_video_arr.npy')
    # meniscus_coords = np.load('./data_output/B1-S60_FULL__20210816_202626/meniscus_coords.npy')
    # segment_coords = np.load('./data_output/B1-S60_FULL__20210816_202626/segment_coords.npy')

    user_input = get_user_input()
    directory_path = mkdir_from_input(user_input)

    zoomed_video_arr = get_tube(user_input['input'])
    np.save(os.path.join(directory_path, 'zoomed_video_arr'), zoomed_video_arr)

    bg_sub = background_subtract(zoomed_video_arr)
    np.save(os.path.join(directory_path, 'bg_sub'), bg_sub)

    start = 2338 - 1  # start frame from
    end = 2681
    zoomed_video_arr = zoomed_video_arr[start:end, :, :]
    bg_sub = bg_sub[start:end, :, :]

    cleaned_bg_sub = clean_bg_sub(bg_sub)
    np.save(os.path.join(directory_path, 'cleaned_bg_sub'), cleaned_bg_sub)

    from multi_video_player import video_player
    video_player(0, bg_sub, cleaned_bg_sub, zoomed_video_arr)

    tongue_maxes = find_tongue_end(cleaned_bg_sub)
    # 20 licks per second is right above the max hummingbird feeding speed, therefore FPS/20 (setting upper bounds)
    tongue_max_frames = find_peaks(tongue_maxes, distance=user_input['fps'] / 20)[0]
    np.save(os.path.join(directory_path, 'tongue_max_frames'), tongue_max_frames)
    print(f"distance = {user_input['fps'] / 20}")
    print('Number of maximums=', len(tongue_max_frames))
    selected_frames = cleaned_bg_sub[tongue_max_frames, :, :]
    selected_color = zoomed_video_arr[tongue_max_frames, :, :]

    meniscus_coords = get_meniscus(selected_frames, selected_color)

    meniscus_arr = update_meniscus_position(meniscus_coords, tongue_max_frames, np.shape(cleaned_bg_sub)[0])
    np.save(os.path.join(directory_path, 'meniscus_arr'), meniscus_arr)

    tongue_pixels = extract_tongue_pixels(cleaned_bg_sub, meniscus_arr, tongue_maxes)
    segment_coords = piecewise_linear(tongue_pixels)
    np.save(os.path.join(directory_path, 'segment_coords'), segment_coords)

    tongue_lengths, just_maxes = analyse_data(user_input, tongue_max_frames, segment_coords, meniscus_arr)

    show_line(bg_sub, False, segment_coords, os.path.join(directory_path, 'line_bw'))
    show_line(zoomed_video_arr, True, segment_coords, os.path.join(directory_path, 'line_color'))
    show_line(tongue_pixels, False, segment_coords, os.path.join(directory_path, 'line_only_tongue'))

    save_results(user_input, directory_path, tongue_lengths, just_maxes)


def update_meniscus_position(meniscus_coords_arr, update_position_frame, total_frame_count):
    meniscus = np.zeros((total_frame_count, 2))
    frame_update_count = 0
    for frame_num in range(total_frame_count):
        if frame_num in update_position_frame:
            meniscus[frame_num, :] = meniscus_coords_arr[frame_update_count, :]
            frame_update_count += 1
        else:
            meniscus[frame_num, :] = meniscus[frame_num - 1, :]
    return meniscus


# Displays the video with the line added on top. Returns a numpy array of the video with the line.
# Optionally saves the video to the output_file_path.
def show_line(video_arr, is_color, segment_coords, output_file_path=None):
    save_arr = []
    if is_color:
        color = (0, 255, 0)
        (frame_height, frame_width, rgb_intensities) = video_arr[0].shape
    else:
        color = (255, 255, 255)
        (frame_height, frame_width) = video_arr[0].shape
    segment_coords = np.round(segment_coords)
    for frame_num, frame in enumerate(video_arr):
        color = (0, 255, 0)
        thickness = 2
        img = frame.copy()
        if not is_color:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        frame_segments = segment_coords[frame_num]
        try:
            point_a = (int(frame_segments[0, 0]), int(frame_segments[1, 0]))
        except ValueError:
            print("********")
            print(f'frame_num:{frame_num}')
            print(f'x_a:{frame_segments[0, 0]}')
            print(f'x_a:{frame_segments[1, 0]}')
            print("********")
            point_a = (-1, -1)
        point_b = (int(frame_segments[0, 1]), int(frame_segments[1, 1]))
        point_c = (int(frame_segments[0, 2]), int(frame_segments[1, 2]))
        cv.line(img, point_a, point_b, color, thickness)
        cv.line(img, point_b, point_c, color, thickness)

        y_dim, x_dim, color_values = np.asarray(np.shape(img))
        # magnification = 3
        # dim = (x_dim * magnification, y_dim * magnification)
        # img = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)

        save_arr.append(img)
        # cv.imshow('frame', img)
        # key = cv.waitKey(100)  # waits 8ms between frames
        # if key == 27:  # if ESC is pressed, exit loop
        #     break
    cv.destroyAllWindows()
    if output_file_path is not None:
        np.save(output_file_path, save_arr)
    return save_arr


# TODO: allow user to set --algo to switch between MOG2 and KNN
# Applies opencv's background subtract method to the inputted video
# and returns that video as an array of frames that contain frame data.
# The input must be an array of frames starting from 0
# learning_rate = -1 means learning rate is set algorithmically,
# learning_rate = 0 means background model is never updated.
def background_subtract(input_video_arr, learning_rate=-1, algo='KNN'):
    # total_frame_count = len(input_video_arr)
    # print(f'Total number of frames: {total_frame_count}')
    print('Starting Background Subtract')
    if algo == 'KNN':
        back_sub = cv.createBackgroundSubtractorKNN(dist2Threshold=100, detectShadows=False)
    elif algo == 'MOG2':
        back_sub = cv.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=20)  # Raise threshold & history?
    else:
        raise ValueError("Unknown background subtract method")
    bg_subbed_vid_arr = []
    # print("Beginning Calibration")
    # for frame in input_video_arr:
    #     back_sub.apply(frame, learningRate=learning_rate)
    # print("Done with Calibration")
    for frame in input_video_arr:
        foreground_mask = back_sub.apply(frame, learningRate=learning_rate)
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


# Configures parameters the user inputs when opening the program
# Returns a dictionary of parameters.
def get_user_input():
    parser = argparse.ArgumentParser(description='Uses computer vision to turn video of birds feeding into data.')
    parser.add_argument('-i', '--input', required=True, help='path to input video')
    parser.add_argument('-f', '--fps', required=True, type=int,
                        help='input the frames per second that the video was captured at')
    parser.add_argument('-o', '--output', help='path to output folder', default='./data_output/')
    # parser.add_argument('-r', '--save_runtime', default=False,
    #                     help='saves runtime files so the exact session can be reproduced')
    args = vars(parser.parse_args())
    args['output'].strip(' ./')
    path = args['input'].strip(' ./')
    if not os.path.isfile(path):
        print('Could not find the file: ', args['input'])
        sys.exit(-1)
    args['input'] = path
    return args


# makes main() a main function similar to Java, C, C++
if __name__ == '__main__':
    main()

########################################################################################
# Maybe stuff:


# for frame in zoomed_video_arr:
#     edges = cv.Canny(frame, 200, 250)
#     # linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
#     # if linesP is not None:
#     #     for i in range(0, len(linesP)):
#     #         l = linesP[i][0]
#     #         cv.line(edges, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
#     cv.imshow('frame', edges)
#
#     # imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#     # ret, thresh = cv.threshold(imgray, 127, 255, 0)
#     # im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     key = cv.waitKey(16)  # waits 8ms between frames
#     if key == 27:  # if ESC is pressed, exit loop
#         break
# view_video(mog_bg_sub, False)


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


# def view_video(video_arr, is_color, *args):
#     if is_color:
#         color = (0, 255, 0)
#         (frame_height, frame_width, rgb_intensities) = video_arr[0].shape
#     else:
#         color = (255, 255, 255)
#         (frame_height, frame_width) = video_arr[0].shape
#
#     line_video_arr = []
#     frame_num = 0
#     for frame in video_arr:
#         # print(frame_num)
#         video_num = 0
#         for video in args:
#             cv.imshow(f'video{video_num}', video[frame_num])
#         cv.imshow('frame', frame)
#         line_video_arr.append(frame)
#         key = cv.waitKey(20)  # waits 8ms between frames
#         if key == 27:  # if ESC is pressed, exit loop
#             break
#         frame_num += 1
#     cv.destroyAllWindows()
#     line_video_arr = np.asarray(line_video_arr)
#     return line_video_arr


# def show_both_loc(tongue_xy_coords, video_to_compare_arr, meniscus_xy_coords):
#     (frame_height, frame_width, rgb_intensities) = video_to_compare_arr[0].shape
#
#     line_video_arr = []
#     frame_num = 0
#     for frame in video_to_compare_arr:
#         # print(frame_num)
#         thickness = 1
#         radius = 1
#
#         for idx, y_coord in enumerate(tongue_xy_coords[frame_num]):
#             color = (0, 255, 0)
#             if np.isnan(y_coord):
#                 y_coord = -1  # replace nans with placeholder
#             center_coordinates = idx, int(y_coord)
#             frame = cv.circle(frame, center_coordinates, radius, color, thickness)
#
#         for y_coord, x_coord in enumerate(meniscus_xy_coords[frame_num]):
#             color = (255, 0, 0)
#             center_coordinates = x_coord, y_coord
#             frame = cv.circle(frame, center_coordinates, radius, color, thickness)
#
#         cv.imshow('frame', frame)
#         line_video_arr.append(frame)
#         key = cv.waitKey(0)  # waits 8ms between frames
#         if key == 27:  # if ESC is pressed, exit loop
#             break
#         frame_num += 1
#     cv.destroyAllWindows()
#     line_video_arr = np.asarray(line_video_arr)
#     return line_video_arr
#
#
# def show_tongue_loc(tongue_xy_coords, video_to_compare_arr):
#     color = (0, 255, 0)
#     (frame_height, frame_width, rgb_intensities) = video_to_compare_arr[0].shape
#
#     line_video_arr = []
#     frame_num = 0
#     for frame in video_to_compare_arr:
#         # print(frame_num)
#
#         thickness = 1
#         radius = 1
#
#         for idx, y_coord in enumerate(tongue_xy_coords[frame_num]):
#             center_coordinates = idx, y_coord
#             frame = cv.circle(frame, center_coordinates, radius, color, thickness)
#
#         cv.imshow('frame', frame)
#         line_video_arr.append(frame)
#         key = cv.waitKey(8)  # waits 8ms between frames
#         if key == 27:  # if ESC is pressed, exit loop
#             break
#         frame_num += 1
#     cv.destroyAllWindows()
#     line_video_arr = np.asarray(line_video_arr)
#     return line_video_arr
#
#
# # Takes the estimated x position and a cropped video array
# # Displays, then Returns a video array with the estimated x-position line
# # included in it. Can include additional inputted positions in the
# # *args.
# def show_position(estimated_position, video_to_compare_arr, is_color, *args):
#     if is_color:
#         color = (0, 255, 0)
#         (frame_height, frame_width, rgb_intensities) = video_to_compare_arr[0].shape
#     else:
#         color = (255, 255, 255)
#         (frame_height, frame_width) = video_to_compare_arr[0].shape
#
#     line_video_arr = []
#     frame_num = 0
#     for frame in video_to_compare_arr:
#         # print(frame_num)
#
#         start_point = estimated_position[frame_num], 0
#         end_point = estimated_position[frame_num], frame_height
#         thickness = 1
#         frame = cv.line(frame, start_point, end_point, color, thickness)
#
#         for arg in args:
#             start_point = arg[frame_num], 0
#             end_point = arg[frame_num], frame_height
#             frame = cv.line(frame, start_point, end_point, color, thickness)
#
#         cv.imshow('frame', frame)
#         line_video_arr.append(frame)
#         key = cv.waitKey(50)  # waits 8ms between frames
#         if key == 27:  # if ESC is pressed, exit loop
#             break
#         frame_num += 1
#     cv.destroyAllWindows()
#     line_video_arr = np.asarray(line_video_arr)
#     return line_video_arr


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
