import numpy as np
from scipy import optimize
# import math
import matplotlib.pyplot as plt
import cv2 as cv


def main():
    # a = np.load('./data_output/mog_bg_sub.npy')
    # a = a[24]
    # rotated = rotate_bound(a, 30)
    # while True:
    #     cv.imshow('un-rotated', a)
    #     cv.imshow('rotated', rotated)
    #     key = cv.waitKey(2)
    a = angle_between((1, 0), (1, -1))
    print(a)

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


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




# from __future__ import print_function
# import numpy as np
# import cv2 as cv
# import argparse
# zoomed_video_arr = np.load('./data_output/cropped_video.npy')
# zoomed_video_arr = zoomed_video_arr[61]
# max_lowThreshold = 100
# window_name = 'Edge Map'
# title_trackbar = 'Min Threshold:'
# ratio = 3
# kernel_size = 3
# def CannyThreshold(val):
#     low_threshold = val
#     img_blur = cv.blur(src_gray, (3,3))
#     detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
#     mask = detected_edges != 0
#     dst = src * (mask[:,:,None].astype(src.dtype))
#     cv.imshow(window_name, dst)
# # parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
# # parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
# # args = parser.parse_args()
# # src = cv.imread(cv.samples.findFile(args.input))
# src = zoomed_video_arr
# if src is None:
#     print('Could not open or find the image: ', args.input)
#     exit(0)
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# cv.namedWindow(window_name)
# cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
# CannyThreshold(0)
# cv.waitKey()


# def main():
#     class CoordinateStore:
#         def __init__(self):
#             self.points = []
#
#         def select_point(self, event, x, y, flags, param):
#             if event == cv2.EVENT_LBUTTONDBLCLK:
#                 cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
#                 self.points.append((x, y))
#
#     # instantiate class
#     coordinateStore1 = CoordinateStore()
#
#     # Create a black image, a window and bind the function to window
#     img = np.zeros((512, 512, 3), np.uint8)
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', coordinateStore1.select_point)
#
#     while True:
#         cv2.imshow('image', img)
#         k = cv2.waitKey(20) & 0xFF
#         if k == 27:
#             break
#     cv2.destroyAllWindows()
#
#     print("Selected Coordinates: ")
#     for i in coordinateStore1.points:
#         print(i)


# https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a

# def main():
#     # meniscus_eq = np.asarray([-1.76432786e-02, 1.23399985e+00, 4.36404085e+01])
#     points = np.load('./data_output/mog_bg_sub.npy')
#     # meniscus = np.load('./data_output/meniscus_coords_arr.npy')
#     meniscus = np.load('./data_output/per_frame_meniscus_coords.npy')
#     # points = np.load('./data_output/tongue_points.npy', allow_pickle=True)
#     max_frames = np.array([24,   64,   97,  141,  174,  209,  241,  272,  307,  343,  377,
#         412,  447,  486,  527,  573,  610,  647,  690,  736,  782,  828,
#         871,  911,  963, 1007, 1047, 1090, 1137, 1182, 1212])
#     for frame_num in max_frames:
#         meniscus = int(meniscus[frame_num][0])
#         print(meniscus)
#         # meniscus_max = np.max(meniscus)
#         points = points[frame_num]
#
#         cv.imshow('frame', points)
#         points = np.asarray(points)
#         print(np.shape(points))
#
#         points[:, :(meniscus + 2)] = 0  # setting lower bounds (+ 2 pixels to remove artifacts due to the meniscus)
#         points[:, 163:] = 0  # setting upper bounds
#         cv.imshow('frame2', points)
#         y, x = points.nonzero()
#
#         # x = np.arange(len(points))
#         # y = points
#         # x = x[40:]
#         # y = points[40:]
#         # print(y)
#
#         px, py = segments_fit(x, y)
#         print('x: ', px)
#         print('y: ', py)
#
#         # solx, soly = intercepts(px, py, meniscus_eq)
#         # print(solx, soly)
#
#         plt.plot(x, y, 'o')
#         plt.plot(px, py, 'or-')
#         # plt.plot([solx], [soly], 'rx')
#         plt.show()
#         key = cv.waitKey(300)  # waits _ms between frames
#         if key == 27:  # if ESC is pressed, exit loop
#             break


def intercepts(x, y, meniscus):
    if y[0] - y[1] == 0:  # Then 1 intercept
        soly = y
        # solx =  I STOPPED WORKING HERE

    m = (x[0] - x[1]) / (y[0] - y[1])  # Reversed so the line is inverted
    intercept = -y[0] * m + x[0]  # Reversed so the line is inverted
    print(m, intercept)

    a = meniscus[0]
    b = meniscus[1] - m
    c = meniscus[2] - intercept

    # calculate the discriminant
    d = (b ** 2) - (4 * a * c)

    # find two solutions
    sol1 = (-b - math.sqrt(d)) / (2 * a)
    sol2 = (-b + math.sqrt(d)) / (2 * a)
    print(sol1)
    soly = m * sol1 + intercept

    return soly, sol1  # Reversed so the number is re-inverted


# Returns two separate x and y variables containing the coordinates
# required for the piecewise linear regression lines. Takes
# an array/list of X and Y coordinates as inputs.
# The number of lines (count) default is 2.
def segments_fit(X, Y, count=2):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2) ** 2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


if __name__ == '__main__':
    main()
