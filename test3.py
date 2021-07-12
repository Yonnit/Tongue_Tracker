import cv2 as cv
import numpy as np
import sys

a = np.asarray([[0, 255],
                [255, 0]], dtype=np.uint8)
print(a)
x_dim, y_dim = np.asarray(np.shape(a))
magnification = 3
dim = (x_dim * magnification, y_dim * magnification)
a = cv.resize(a, dim, interpolation=cv.INTER_NEAREST)
cv.imshow('frame', a)
while True:
    cv.imshow('frame', a)
    key = cv.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):  # If the user presses Q or ESC
        cv.destroyAllWindows()
        sys.exit(-1)
cv.destroyAllWindows()


