import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard

# Using built-in face cascade for now
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Must define viable camera (troubleshoot video0-4 if not working)
cap = cv2.VideoCapture("/dev/video0")
# Stops code in the case of failure
if not cap.isOpened():
        print("Error: Couldn't open the camera.")
    

def main():

    while True:
        frame = readim(cap)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

# Take in image from camera
def readim(capchannel):
    # Read in the frame
    ret, frame = capchannel.read()

    # Check if the frame failed to read
    if not ret:
        print("Error: Couldn't read frame.")

    # If successfully read frame, return it
    else:
        return frame

# Run cascae (specify both image and cascade)
def cascade(cascade, im):

    # Obtain grayscale
    im = grayscale(im)

    # Perform face detection
    objects = cascade.detectMultiScale(im, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

# Finds the coordinate of the center of the object
def center(tl_point, width):

    pass

# Calculates the magnitude and direction of adjustment necessary
def calc_pos(center, width):

    pass

# This calculates the real necessary position adjustment to sevor positions necessary
def calc_adjust(x, y, size):

    pass

# This pushes the position adjustments to the servos (actually moves them to track detected objects)
def pos_adjust(dx, dy, magnitude):

    pass

# Camera traces out available positions - this behavior breaks when an object is spotted
def sweep():

    pass

# Converts a frame to grayscale
def grayscale(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Return Grayscale
    return gray 

# Applies a gaussian blur to a frame
def blur():

    pass

# Runs canny edge detection on a frame
def canny():

    pass

# Runs main function
if __name__ == "__main__":
    main()