import numpy as np
import cv2 as cv
import sys


# Returns the coordinates of the points where the meniscus intersects with the tongue.
# The user selects the points from the inputted video (array of images)
# The inputted video must be a numpy array and can be black and white or color
def get_meniscus(selected_frames, selected_color):
    # selected_frames = np.load('./data_output/selected_frames.npy')
    meniscus_coords = []
    frame_num = 0
    magnification = 1
    is_color = False
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
        print("Press '+' to zoom in. Press '-' to zoom out")
        if not is_color:
            frame = selected_frames[frame_num, :, :]
        else:
            frame = selected_color[frame_num, :, :]
        coords, magnification, is_color = select_intersection(frame, magnification, is_color)
        print(coords)
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
def select_intersection(frame, magnification, is_color):
    class CoordinateStore:
        def __init__(self, magnification):
            self.points = -1
            self.clicked = False
            self.magnification = magnification

        def select_point(self, event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
                cv.circle(img, (x, y), 3, (0, 120, 0), -1)
                self.points = (int(x / self.magnification), int(y / self.magnification))
                self.clicked = True

        def zoom_in(self):
            self.magnification += 1

        def zoom_out(self):
            if self.magnification > 1:
                self.magnification -= 1

    # instantiate class
    coordinateStore1 = CoordinateStore(magnification)

    # Creates a copy of the image so the array isn't changed
    img = frame.copy()
    if not is_color:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.namedWindow('image')
    cv.setMouseCallback('image', coordinateStore1.select_point)

    y_dim, x_dim, px_value = np.asarray(np.shape(img))

    while True:
        magnification = coordinateStore1.magnification
        dim = (x_dim * magnification, y_dim * magnification)
        img = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == 27 or k == ord("q"):  # ESC or Q
            cv.destroyAllWindows()
            sys.exit(-1)
        elif k == 32:  # SPACE
            cv.destroyAllWindows()
            return coordinateStore1.points, magnification, is_color
        elif k == ord('r'):  # R
            cv.destroyAllWindows()
            return -1, magnification, is_color
        elif k == ord('s'):
            cv.destroyAllWindows()
            return -1, magnification, not is_color
        elif k == ord('='):
            coordinateStore1.zoom_in()
        elif (k == ord("-")) and coordinateStore1.magnification > 1:
            coordinateStore1.zoom_out()


if __name__ == '__main__':
    main()
