import numpy as np
import cv2

# This code was written by Adrian Rosebrock and can be found at this link:
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# I made some minor modifications to the order_points function because it did not account for
# all cases leading to bugs


# pts are the coordinates selected by the user
# Re-orders the points so that each corner clicked populates the
# same vertical index of the coordinate array, which is then returned
# TODO: you might be able to rotate/flip the box by switching which coordinates go where
def order_points(pts):
    # Initialize a 4x2 array to be populated with coords
    # 0 = top left, 1 = top right, 2, = bottom right, 3 = bottom left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    array_min = np.argmin(s)
    rect[0] = pts[array_min]
    pts = np.delete(pts, array_min, 0)

    s = pts.sum(axis=1)  # update the s array
    array_max = np.argmax(s)
    rect[2] = pts[array_max]
    pts = np.delete(pts, array_max, 0)

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


# Receives coordinate points, returns data needed to apply warp
def get_four_point_transform(pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    transform_data = (M, maxWidth, maxHeight)
    return transform_data


# Apply the four point transform.
# Takes the getPerspectiveTransform object as input
def apply_four_point_transform(image, transform_data):
    (M, maxWidth, maxHeight) = transform_data
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

