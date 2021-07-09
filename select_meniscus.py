import numpy as np
import cv2 as cv
import sys


# Returns the coordinates of the points where the meniscus intersects with the tongue.
# The user selects the points from the inputted video (array of images)
# The inputted video must be a numpy array and can be black and white or color
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
        if coords != -1:
            meniscus_coords.append(coords)
            frame_num += 1

    meniscus_coords = np.asarray(meniscus_coords)
    return meniscus_coords


# Returns the coordinates of a user's first click. Also draws a circle under the clicked point
# so the user can assess whether they want to redo or continue.
# Takes an image as input, along with user mouse and keyboard input
# If the user doesn't click, or wants to redo ('R'), the function returns -1
# If the user presses Q or ESC, the program closes.
# If the user clicks and presses SPACE, the coordinates are returned as tuples (x, y)
def select_intersection(frame):
    class CoordinateStore:
        def __init__(self):
            self.points = -1
            self.clicked = False

        def select_point(self, event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
                cv.circle(img, (x, y), 3, (0, 120, 0), -1)
                self.points = (x, y)
                self.clicked = True

    # instantiate class
    coordinateStore1 = CoordinateStore()

    # Creates a copy of the image so the array isn't changed
    img = frame.copy()
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
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


if __name__ == '__main__':
    main()
