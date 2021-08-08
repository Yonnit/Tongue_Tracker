import cv2 as cv
import numpy as np
import sys


def main():
    path = './data_output/B2-S20__20210805_224343'
    a = np.load(f'{path}/bw_line.npy')
    a2 = np.load(f'{path}/color_line.npy')
    a3 = np.load(f'{path}/only_tongue_line.npy')
    video_player(682, a, a2, a3)


def video_player(starting_frame, *args):
    try:
        total_frames, y_dim, x_dim, px_value = np.asarray(np.shape(args[1]))
    except ValueError:
        total_frames, y_dim, x_dim = np.asarray(np.shape(args[1]))  # For single channel images (black and white)

    print(f'Starting on frame {starting_frame} of {total_frames}')
    magnification = 1
    frame_num = starting_frame
    while True:
        print(frame_num)
        dim = (x_dim * magnification, y_dim * magnification)
        for i, video in enumerate(args):
            frame = cv.resize(video[frame_num], dim, interpolation=cv.INTER_NEAREST)
            cv.imshow(f'video_{i}', frame)
        key = cv.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            break
        # elif key in {ord('k'), 32}:  # key is 'SPACE' or 'k'
        #     if ms_per_frame == 0:
        #         ms_per_frame = 8
        #     else:
        #         ms_per_frame = 0
        elif key == ord('j'):
            frame_num -= 1
        elif key == ord('l'):
            frame_num += 1
        elif key == ord('h'):
            frame_num -= 2
        elif key == ord(';'):
            frame_num += 2
        elif key in {ord('='), ord("+")}:
            magnification += 1
        elif (key == ord("-")) and magnification > 1:
            magnification -= 1
    cv.destroyAllWindows()
    sys.exit(-1)


if __name__ == '__main__':
    main()
