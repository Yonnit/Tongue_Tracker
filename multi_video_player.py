import cv2 as cv
import numpy as np
import sys


def video_player(starting_frame, *args):
    try:
        total_frames, y_dim, x_dim, px_value = np.asarray(np.shape(args[0]))
    except ValueError:
        total_frames, y_dim, x_dim = np.asarray(np.shape(args[0]))  # For single channel images (black and white)

    print(f'Starting on frame {starting_frame} of {total_frames} total frames')
    magnification = 1
    frame_num = starting_frame
    while True:
        frame_num = frame_num % total_frames
        print(frame_num)
        dim = (x_dim * magnification, y_dim * magnification)
        for i, video in enumerate(args):
            frame = cv.resize(video[frame_num], dim, interpolation=cv.INTER_NEAREST)
            cv.imshow(f'video_{i}', frame)
        key = cv.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):  # If the user presses Q or ESC
            cv.destroyAllWindows()
            sys.exit(-1)
        elif key == ord(" "):
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


def main():
    videos = get_user_input()
    video_player(0, *videos)


# Configures parameters the user inputs when opening the program
# Returns a dictionary of parameters.
def get_user_input():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='A video player that can play one or more videos simultaneously.')
    parser.add_argument('-f', '--folder', required=True, help='path to folder that files are in. If list_input'
                                                              'is left empty, will automatically try to open'
                                                              'default output video files from tongue_tracking')
    parser.add_argument('-li', '--list_input', nargs='+', help='list of file names for the video player to open')
    args = vars(parser.parse_args())
    args['folder'] = args['folder'].strip(' ./')
    if not os.path.isfile(args['folder']):
        raise FileNotFoundError(f"Could not find the folder: {args['folder']}")
    if args['li'] is None:
        path = args['folder']
        args['li'] = f'{path}/line_bw.npy', f'{path}/line_color.npy', f'{path}/line_only_tongue.npy', \
                     f'{path}/zoomed_video_arr.npy'
    video_list = []
    for arg in args['li']:
        video_list.append(np.load(arg))
    return video_list


if __name__ == '__main__':
    main()
