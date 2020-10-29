import numpy as np
import cv2 as cv
import sys

from transform import get_four_point_transform, apply_four_point_transform


# Gets tube corners and returns video array zoomed in on the tube
def get_tube(file_path):  # TODO: make filepath user inputted
    first_frame = grab_first_frame(file_path)
    tube_location_data = select_corners(first_frame)
    vid_tube = zoom_into_tube(tube_location_data, file_path)
    return vid_tube


# Returns an image of the first frame of the video inputted
def grab_first_frame(input_path):
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(-1)
    (exists_frame, frame) = cap.read()
    cap.release()
    return frame


# TODO: If you want, make the image global so that you can put in
#   circles in as the user clicks
# Handles mouse clicks in the select corners function
def click_and_crop(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN) & (len(CORNER_COORDS) < 4):
        click_coords = (x, y)
        CORNER_COORDS.append(click_coords)
        print(f'Clicked corner {len(CORNER_COORDS)}/4')


# TODO: somehow figure out how to deal with different feeding tube orientations
#   somewhere the user is going to have to input which way the tube is facing
#   can be dealt with during the to vertical bars portion or manually rotated here
# Displays image, prompts user to click 4 corners of desired object.
# Returns the data which can later be used to zoom into the area contained
# within the four points
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
    corner_array = np.array(CORNER_COORDS, dtype="float32")
    transform_data = get_four_point_transform(corner_array)
    warped = apply_four_point_transform(img, transform_data)

    print("If unsatisfied with the crop, press 'Q' to Cancel")
    print("Otherwise, press 'Y' to continue")
    while True:
        cv.imshow("Cropped image. Press 'Y' to Continue", warped)
        print(f"The dimensions of the cropped image: Height={warped.shape[0]}, Width={warped.shape[1]}")
        key = cv.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
        elif key == ord("y"):
            break
    cv.destroyAllWindows()
    return transform_data


# Takes the corner coordinate data and zooms the video into the coordinate
# data collected. Returns an array which contains the frame data with the
# index being the frame number starting from 0.
def zoom_into_tube(transform_data, file_path):
    cap = cv.VideoCapture(file_path)
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(-1)
    total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames: {total_frame_count}')
    print("Zooming into cropped region")
    video_array = []  # array of frames, index = frame # starting from 0

    while True:
        (exists_frame, frame) = cap.read()

        if not exists_frame:
            break
        cropped_frame = apply_four_point_transform(frame, transform_data)
        video_array.append(cropped_frame)
    cap.release()
    cv.destroyAllWindows()

    video_array = np.asarray(video_array)
    print("Done with zooming")
    return video_array
