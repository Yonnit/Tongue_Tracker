import numpy as np
import cv2 as cv
import argparse
import sys

from transform import get_four_point_transform, apply_four_point_transform


def main():
    input_path = file_and_path()
    first_frame = grab_first_frame(input_path)
    select_corners(first_frame)

    # bg_sub_array = background_subtract()
    # to_vertical_bands(bg_sub_array)


# Returns an image of the first frame of the video inputted
def grab_first_frame(input_path):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(-1)
    (exists_frame, frame) = cap.read()
    cap.release()
    return frame


# Displays image, prompts user to click 4 corners of image.
def select_corners(img):
    global CORNER_COORDS  # double check that I actually need a global variable here
    CORNER_COORDS = []

    win_name = "Click the 4 Corners (Press 'Q' to Cancel)"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, click_and_crop)

    # While less than 4 corners are clicked
    while len(CORNER_COORDS) < 4:
        cv.imshow(win_name, img)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
    cv.destroyAllWindows()
    print(CORNER_COORDS)
    corner_array = np.array(CORNER_COORDS, dtype="float32")
    transform_data = get_four_point_transform(corner_array)
    warped = apply_four_point_transform(img, transform_data)

    print('If unsatisfied with the crop, press "Q" to Cancel')
    while True:
        cv.imshow('Cropped image. Press "Y" to Continue', warped)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
        elif key == ord("y"):
            break
    return transform_data


# TODO: If you want, make the image global so that you can put in
#   circles in as the user clicks
def click_and_crop(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN) & (len(CORNER_COORDS) < 4):
        click_coords = (x, y)
        CORNER_COORDS.append(click_coords)
        print(f'Clicked corner {len(CORNER_COORDS)}/4')


# Returns the average brightness of a vertical slice of pixels
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def to_vertical_bands(input_array):
    # a = np.zeros(len(input_array))
    input_array = input_array.mean(axis=1)
    print(f'Frame count, Horizontal Resolution: {input_array.shape}')
    np.savetxt('./data_output/foo.csv', input_array, delimiter=',')


# Simple background subtraction
# Saves video to output file
# Returns a 3d numpy matrix containing the video information
# TODO: allow user to set --algo to switch between MOG2 and KNN
# TODO: Make saving video to output file optional
# TODO: When next testing background sub method, remember that you changed your mind
#   so file_and_path should be inputted as a parameter instead of directly into file
# TODO: Remember that bg_sub should no longer display frame, separate out into diff
#   methods like watch_video, save_video and such
def background_subtract():
    back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30, detectShadows=False)
    cap = cv.VideoCapture(file_and_path())
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(-1)
    # Obtains default resolution of frame & converts from float to int
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'Frame resolution: Width={frame_width} Height={frame_height}')
    print(f'Total number of frames: {total_frame_count}')
    print('Press Q to quit')
    # Define the codec, create VideoWriter object.
    output = cv.VideoWriter('./data_output/Background_Sub.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            3, (frame_width, frame_height), 0)
    video_array = []  # array of frames, index = frame # starting from 0
    while True:
        (exists_frame, frame) = cap.read()

        if not exists_frame:
            break
        fg_mask = back_sub.apply(frame)
        # Puts the frame count on the original video
        # frame_number starts from 1
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        video_array.append(fg_mask)
        output.write(fg_mask)
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fg_mask)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    output.release()
    cv.destroyAllWindows()

    video_array = np.asarray(video_array)
    return video_array


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