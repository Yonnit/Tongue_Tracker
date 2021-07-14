import cv2 as cv
import numpy as np
import sys

# a = np.asarray([[0, 255],
#                 [255, 0]], dtype=np.uint8)
a = np.load('./data_output/line_vid_color.npy')
a2 = np.load('./data_output/line_vid.npy')
a3 = np.load('./data_output/line_vid_tongue.npy')
# frame_num = 413
# a2 = a2[frame_num]
#
# a = a[frame_num]
print(a)
ignore, y_dim, x_dim, px_value = np.asarray(np.shape(a))
magnification = 3
dim = (x_dim * magnification, y_dim * magnification)
frame_num = 412
while True:
    print(frame_num)
    frame = cv.resize(a[frame_num], dim, interpolation=cv.INTER_NEAREST)
    frame2 = cv.resize(a2[frame_num], dim, interpolation=cv.INTER_NEAREST)
    frame3 = cv.resize(a3[frame_num], dim, interpolation=cv.INTER_NEAREST)
    cv.imshow('frame', frame)
    cv.imshow('frame2', frame2)
    cv.imshow('frame3', frame3)
    key = cv.waitKey(0) & 0xFF
    if key == 27 or key == ord("q"):  # If the user presses Q or ESC
        cv.destroyAllWindows()
        sys.exit(-1)
    elif key == ord('j'):
        frame_num -= 1
    elif key == ord('l'):
        frame_num += 1
cv.destroyAllWindows()


