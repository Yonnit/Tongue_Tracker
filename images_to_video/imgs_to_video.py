import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--file", required=True, help="name of file containing images to convert")
# ap.add_argument("-ext", "--extension", required=False, default='bmp', help="extension name. default is 'bmp'.")
# ap.add_argument("-o", "--output", required=False, default='vid_from_imgs.mp4', help="output video file")
# args = vars(ap.parse_args())

# # Arguments
# dir_path = '.' + args['file']
# ext = args['extension']
# output = args['output']
dir_path = "Export_20171116_033705_PM"
ext = "bmp"
output = f"{dir_path}.avi"

images = []
print(dir_path)
print(os.listdir(dir_path))
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video', frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # can also use (*'mp4v') if we want mp4 (may make smaller files)
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))