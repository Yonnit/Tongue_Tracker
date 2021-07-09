import numpy as np
import cv2 as cv
import sys


def get_meniscus(selected_frames):
    # selected_frames = np.load('./data_output/selected_frames.npy')

    meniscus_coords = []
    frame_num = 0
    total_frames = np.shape(selected_frames)[0]

    print()
    print('===================')
    print('You will now select the point at which the tongue intersects with the meniscus')
    print(f'There are {total_frames} images to select')
    print()

    while frame_num < total_frames:
        print(f'{frame_num + 1}/{total_frames}')
        print("Click the point at which the tongue intersects with the meniscus, then")
        print("Press 'SPACE' to continue, 'R' to redo, 'Q' to quit ")
        frame = selected_frames[frame_num, :, :]
        coords = select_intersection(frame)
        if coords is not -1:
            meniscus_coords.append(coords)
            frame_num += 1

    meniscus_coords = np.asarray(meniscus_coords)
    return meniscus_coords


def select_intersection(frame):
    class CoordinateStore:
        def __init__(self):
            self.points = -1
            self.clicked = False

        def select_point(self, event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
                cv.circle(img, (x, y), 2, (0, 255, 0), -1)
                self.points = (x, y)
                self.clicked = True

    # instantiate class
    coordinateStore1 = CoordinateStore()

    # Create a black image, a window and bind the function to window
    img = frame.copy()
    cv.namedWindow('image')
    cv.setMouseCallback('image', coordinateStore1.select_point)

    while True:
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == 27 or k == ord("q"):  # ESC or Q
            cv.destroyAllWindows()
            sys.exit(-1)
        elif k == 32:  # SPACE
            cv.destroyAllWindows()
            return coordinateStore1.points
        elif k == ord('r'):  # R
            cv.destroyAllWindows()
            return -1


    print("Selected Coordinates: ")
    # for i in coordinateStore1.points:
    print(coordinateStore1.points)






# # Handles mouse clicks
# def click_and_crop(event, x, y, flags, param):
#     if event == cv.EVENT_LBUTTONDOWN:
#         click_coords = (x, y)
#         MENISCUS_COORDS.append(click_coords)
#         print(f'Clicked corner {len(MENISCUS_COORDS)}/4')


# Displays image, prompts user to click 4 corners of desired object.
# Returns the data which can later be used to zoom into the area contained
# within the four points
def select_corners(img):
    global MENISCUS_COORDS  # double check that I actually need a global variable here
    MENISCUS_COORDS = []

    win_name = "Click the 4 Corners (Press 'Q' to Cancel)"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, click_and_crop)

    # While less than 4 corners are clicked
    while len(MENISCUS_COORDS) < 4:
        cv.imshow(win_name, img)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
    cv.destroyAllWindows()
    corner_array = np.array(MENISCUS_COORDS, dtype="float32")
    transform_data = get_four_point_transform(corner_array)
    warped = apply_four_point_transform(img, transform_data)

    print(f"The dimensions of the cropped image: Height={warped.shape[0]}, Width={warped.shape[1]}")
    print()
    print("If UNSATISFIED with the crop, press 'Q' to Cancel")
    print("If SATISFIED with the crop, please select the direction of the opening of the tube to continue.")
    print("'L' if the opening is on the left, and 'R' if the opening is on the right")
    while True:
        cv.imshow("Cropped image. Read console for instructions", warped)
        key = cv.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
        elif key == ord("l"):
            opening_is_left = True
            break
        elif key == ord("r"):
            opening_is_left = False
            break
    cv.destroyAllWindows()
    return transform_data, opening_is_left


if __name__ == '__main__':
    main()
