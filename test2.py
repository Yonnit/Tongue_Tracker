import numpy as np
import cv2 as cv
import sys


def main():
    a = np.full((5, 5), 255, np.uint8)
    print(a)
    point = np.array((2, 2))
    x = point[0]
    y = point[1]
    y_dim, x_dim = np.shape(a)
    #
    # print(point - dimensions)
    # k1, k2 = point-dimensions

    a = np.triu(a, x - y)
    print('cut!')
    print(a)
    a = np.flipud(a)
    print('flip!')
    print(a)
    a = np.triu(a, -1 * (y_dim - 1 - y - x))
    print('cut2!')
    print(a)
    a = np.flipud(a)
    print('flip2!')
    print(a)
    # while True:
    #     cv.imshow('frame', a)
    #     key = cv.waitKey(1) & 0xFF
    #     if key == 27 or key == ord("q"):  # If the user presses Q or ESC
    #         cv.destroyAllWindows()
    #         sys.exit(-1)
    # cv.destroyAllWindows()



if __name__ == '__main__':
    main()
