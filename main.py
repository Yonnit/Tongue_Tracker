import numpy as np
import cv2 as cv
import argparse
import sys


def main():
    background_subtract()


# Takes individual frames
# Returns an array whose index represents frame number (starting from 0)
# and contains the picture in its array form.
# def frame_to_array(input_frame, total_frame_count, frame_number):
#     frame_array = np.zeros(total_frame_count)
#     frame_array[frame_number - 1] = input_frame
#     print(frame_array[0])
#     print(frame_array[total_frame_count - 1])

# Simple background subtraction
# Saves video to output file
# TODO: allow user to set --algo to switch between MOG2 and KNN
# TODO: allow user to select output location, then MUST CONFIGURE frame processing() input
def background_subtract():
    back_sub = cv.createBackgroundSubtractorMOG2(varThreshold=30, detectShadows=False)
    cap = cv.VideoCapture(file_and_path())
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(-1)
    # Obtains default resolution of frame & converts from float to int
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'Frame resolution: Width={frame_width} Height={frame_height}')
    print(f'Total number of frames: {total_frame_count}')
    print('Press Q to quit')
    # Define the codec, create VideoWriter object.
    output = cv.VideoWriter('./data_output/Background_Sub.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            10, (frame_width, frame_height), 0)
    video_array = [] # array of frames, index = frame # starting from 0
    while True:
        (exists_frame, frame) = cap.read()

        if not exists_frame:
            break
        fg_mask = back_sub.apply(frame)
        # Puts the frame count on the original video
        # frame_number starts from 1
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        video_array.append(fg_mask)
        output.write(fg_mask)
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fg_mask)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    output.release()
    cv.destroyAllWindows()
    sys.exit(0)


# for now just selects first frame NOTE: BADLY WRITTEN
def select_first_frame():
    video_path = file_and_path()
    cap = cv.VideoCapture(video_path)

    # Checks if file opened
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(-1)

    (exists_frame, frame) = cap.read()
    while True:
        cv.imshow('Frame', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    print("Success...?")


def crop_to_tube():
    video_path = file_and_path()
    cap = cv.VideoCapture(video_path)

    # Checks if file opened
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(-1)

    cv.namedWindow('Frame', cv.WINDOW_AUTOSIZE)

    # Read until end of video
    while cap.isOpened():
        (exists_frame, frame) = cap.read()

        # Displays resulting frame
        if exists_frame:
            cv.imshow('Frame', frame)

            # Press Q to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()
    print("Success...?")

    # Define codec


#    out = cv.VideoWriter('')


# Prompts user to input file name
# Returns path and name in directory
# TODO: configure argparse so user selects file location, and other params
def file_and_path():  # Will likely use argparse here
    input_file_name = "Sequence 01.avi"
    path = "./video_input/"
    return path + input_file_name


# makes main() a main function similar to Java, C, C++
if __name__ == '__main__':
    main()
