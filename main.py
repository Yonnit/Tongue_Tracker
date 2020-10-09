import numpy as np
import cv2 as cv
import argparse
import sys


def main():
    corner_coords = find_tube_coords()
    # bg_sub_array = background_subtract()
    # to_vertical_bands(bg_sub_array)

# Finds edges of feeding tube, locates the corners,
# Returns an object with the coordinates of the corners
# and the side the opening is on
# TODO: make this into an object, so that the inputted video
def find_tube_coords():
    # Pulls first frame
    video_path = file_and_path()
    cap = cv.VideoCapture(video_path)

    # Checks if file opened
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(-1)

    # These following lines of code are "Otsu's method" for automatic thresholding,
    # TODO: Implement automatic thresholding
    # (high_thresh, thresh_im) = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # lowThresh = 0.5 * high_thresh

    (exists_frame, frame) = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Displays frame until user cancels with Q
    print('Press Q to quit')
    while True:
        cv.imshow('gray', gray)
        cv.imshow('edges', edges)
        cv.imshow('lines', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break


# Returns the average brightness of a vertical slice of pixels
# Index represents frame starting from 0
# Within the frame element [y,x] is the pixel location
# ie. second frame y=240 x=600: [1][240, 600] <-- returns intensity
def to_vertical_bands(input_array):
    # a = np.zeros(len(input_array))
    input_array = input_array.mean(axis=1)
    print(f'Frame count, Horizontal Resolution: {input_array.shape}')
    np.savetxt('./data_output/foo.csv', input_array, delimiter=',')


# Simple background subtraction
# Saves video to output file
# Returns a 3d numpy matrix containing the video information
# TODO: allow user to set --algo to switch between MOG2 and KNN
# TODO: Make saving video to output file optional
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
                            3, (frame_width, frame_height), 0)
    video_array = []  # array of frames, index = frame # starting from 0
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

    video_array = np.asarray(video_array)
    return video_array


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
    input_file_name = "Export_20171211_015532_PM.avi"
    path = "./video_input/"
    return path + input_file_name


# makes main() a main function similar to Java, C, C++
if __name__ == '__main__':
    main()
