import numpy as np
import cv2 as cv
import sys

from transform import get_four_point_transform, apply_four_point_transform


# Gets tube corners and returns video array zoomed in on the tube
def get_tube(file_path):  # TODO: make filepath user inputted
    first_frame = grab_first_frame(file_path)
    cropped_frame, roi = select_roi(first_frame)
    while True:
        tube_location_data, key = select_corners(cropped_frame)
        if key != ord("c"):
            break
    vid_tube = zoom_into_tube(tube_location_data, file_path, roi, key)
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


def select_roi(frame):
    print("Place a box around the region of interest")
    print("Press 'SPACE' to continue")
    r = cv.selectROI(frame, fromCenter=False)
    cropped = crop_img(frame, r)
    cv.destroyAllWindows()
    return cropped, r


def crop_img(frame, roi):
    r = roi
    cropped = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    return cropped


# TODO: If you want, make the image global so that you can put in
#   circles in as the user clicks
# Handles mouse clicks in the select corners function
def click_and_crop(event, x, y, flags, param):
    if (event == cv.EVENT_LBUTTONDOWN) & (len(CORNER_COORDS) < 4):
        click_coords = (x/MAGNIFICATION, y/MAGNIFICATION)
        CORNER_COORDS.append(click_coords)
        print(f'Clicked corner {len(CORNER_COORDS)}/4')
        print(click_coords)


# Displays image, prompts user to click 4 corners of desired object.
# Returns the data which can later be used to zoom into the area contained
# within the four points
def select_corners(img):
    global CORNER_COORDS, MAGNIFICATION  # double check that I actually need a global variable here
    CORNER_COORDS = []

    print("Click the 4 Corners of the Tube. Press 'ESC' to Quit")
    win_name = "Click the 4 Corners (Press 'ESC' to Quit)"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, click_and_crop)

    y_dim, x_dim, px_value = np.asarray(np.shape(img))
    MAGNIFICATION = 1
    # While less than 4 corners are clicked
    while len(CORNER_COORDS) < 4:
        dim = (x_dim * MAGNIFICATION, y_dim * MAGNIFICATION)
        frame = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
        cv.imshow(win_name, frame)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
        elif key == ord('='):
            MAGNIFICATION += 1
        elif (key == ord("-")) and MAGNIFICATION > 1:
            MAGNIFICATION -= 1

    cv.destroyAllWindows()
    corner_array = np.array(CORNER_COORDS, dtype="float32")
    transform_data = get_four_point_transform(corner_array)
    warped = apply_four_point_transform(img, transform_data)

    print(f"The dimensions of the cropped image: Height={warped.shape[0]}, Width={warped.shape[1]}")
    print()
    print()
    print("To EXIT the program, press 'ESC'")
    print("If UNSATISFIED with the crop, press 'C' to Cancel and Restart")
    print("If SATISFIED with the crop, please select the direction of the opening of the tube to continue.")
    print("'W' if the opening is on the top, 'S' if the opening is on the bottom")
    print("'A' if the opening is on the left, and 'D' if the opening is on the right")
    while True:
        cv.imshow("Cropped image. Read console for instructions", warped)
        key = cv.waitKey(4) & 0xFF
        if key == 27:
            cv.destroyAllWindows()
            sys.exit(-1)
        elif key == ord("c"):
            print("Cancel")
            break
        elif key in {ord("w"), ord("a"), ord("s"), ord("d")}:
            print("Continue")
            break
    cv.destroyAllWindows()
    return transform_data, key


# Takes the corner coordinate data and zooms the video into the coordinate
# data collected. Returns an array which contains the frame data with the
# index being the frame number starting from 0.
# Opening dir is a integer representing a unicode character.
# The letter indicates the direction of the opening (W,A,S,D)
def zoom_into_tube(transform_data, file_path, roi, opening_dir):
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
        frame = crop_img(frame, roi)
        cropped_frame = apply_four_point_transform(frame, transform_data)
        if opening_dir == ord("d"):
            cropped_frame = cv.flip(cropped_frame, 1)
        elif opening_dir == ord("w"):
            cropped_frame = cv.rotate(cropped_frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        elif opening_dir == ord("s"):
            cropped_frame = cv.rotate(cropped_frame, cv.ROTATE_90_CLOCKWISE)
        video_array.append(cropped_frame)
    cap.release()
    cv.destroyAllWindows()

    video_array = np.asarray(video_array)
    print("Done with zooming")
    print()
    return video_array
