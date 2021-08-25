import numpy as np
import cv2 as cv
import sys
import math

from transform import get_four_point_transform, apply_four_point_transform


# Gets tube corners and returns video array zoomed in on the tube
def get_tube(user_input):
    if 'zoomed' in user_input:
        print(f'Loading zoomed video array from {user_input["zoomed"]}')
        return np.load(user_input['zoomed'])

    file_path = user_input['input']
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
    global CORNER_COORDS, MAGNIFICATION  # I know this is bad practice, blame the person before me
    CORNER_COORDS = []

    print("First, select the two corners on the same side of the tube."
          "Then select any point on the opposite side of the tube."
          ""
          "Press '+' to zoom in. Press '-' to zoom out"
          "Press 'ESC' to Quit")
    win_name = "Follow Console Instructions (Press 'ESC' to Quit)"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, click_and_crop)

    y_dim, x_dim, px_value = np.asarray(np.shape(img))
    MAGNIFICATION = 1
    # While less than 3 clicks
    while len(CORNER_COORDS) < 3:
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

    line_ab = LineSegment(CORNER_COORDS[0], CORNER_COORDS[1])  # line_ab is the side of the tube first selected
    perp_slope = line_ab.get_perpendicular()  # Gets slope of line perpendicular to ab
    x_c, y_c = CORNER_COORDS[2]
    perp_b = -(perp_slope * x_c) + y_c  # Finds b (y-intercept) for the perpendicular line that intersects point c
    intercept = line_ab.get_intercept(perp_slope, perp_b)  # finds where the perpendicular line intercepts line_ab
    line_cd = LineSegment(CORNER_COORDS[2], intercept)  # creates a LineSegment given point_c and the intercept point
    translation_vector = line_cd.get_vector()
    line_gh = line_ab.translate_copy(translation_vector)
    corner_array = line_ab.point_a, line_ab.point_b, line_gh.point_a, line_gh.point_b

    corner_array = np.array(corner_array, dtype="float32")
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

# # Takes 2 corner coords (a and b) and the opposite side's coords
# # Returns the 4 corner coords of the rectangle
# def find_corners(point_a, point_b, point_c):
#     x_a = point_a[0]
#     x_b = point_b[0]
#     y_a = point_a[1]
#     y_b = point_b[1]
#     c_a = point_c[0]
#     c_b = point_c[1]
#     m_ab = (y_a - y_b) / (x_a - x_b)
#     m_cd = -1 / m_ab
#     b_cd =
#     x_d = ()


class LineSegment:
    def __init__(self, point_a, point_b):
        self.point_a = point_a
        self.point_b = point_b
        self.x_a = point_a[0]
        self.x_b = point_b[0]
        self.y_a = point_a[1]
        self.y_b = point_b[1]
        self.m = (self.y_a - self.y_b) / (self.x_a - self.x_b)
        self.b = -self.m * self.x_a + self.y_a

    @staticmethod
    def from_point_slope(point, slope):
        b = -slope * point[0] + point[1]

    def get_intercept(self, m, b):
        xi = (self.b - b) / (m - self.m)
        yi = (self.m * xi + self.b)
        return xi, yi

    def get_perpendicular(self):
        return -1 / self.m

    def get_vector(self):
        return self.x_a - self.x_b, self.y_a - self.y_b

    def translate_copy(self, translation_vector):
        x = translation_vector[0]
        y = translation_vector[1]
        x_a = self.x_a + x
        x_b = self.x_b + x
        y_a = self.y_a + y
        y_b = self.y_b + y
        return LineSegment((x_a, y_a), (x_b, y_b))


# Explanation here: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


# Finds the angle that the line segment connecting the two
# points deviates from horizontal
# Returned value is in degrees
def angle_from_hor(p1, p2):
    theta = math.atan((p2[1] - p1[1])/(p2[0] - p1[0]))
    return np.rad2deg(theta)

