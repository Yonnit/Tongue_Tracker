import numpy as np
import cv2 as cv
import argparse
import sys


def main():
    background_subtract()


# Simple background subtraction
# Saves video to output file
def background_subtract():
    back_sub = cv.createBackgroundSubtractorMOG2()
    cap = cv.VideoCapture(file_and_path())
    if not cap.isOpened():
        print('Error opening video file')
        sys.exit(-1)
    # Obtains default resolution of frame & converts from float to int
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(int(cap.get(4)))
    # Define the codec, create VideoWriter object.
    output = cv.VideoWriter('./data_output/Background_Sub.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            10, (frame_width,frame_height), 0)
    while True:
        (exists_frame, frame) = cap.read()

        if not exists_frame:
            break
        fg_mask = back_sub.apply(frame)
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        output.write(fg_mask)
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fg_mask)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    output.release()
    cv.destroyAllWindows()

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
def file_and_path():  # Will likely use argparse here
    input_file_name = "Sequence 01.avi"
    path = "./video_input/"
    return path + input_file_name


# makes main() a main function similar to Java, C, C++
if __name__ == '__main__':
    main()
