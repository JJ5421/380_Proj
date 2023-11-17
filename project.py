import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard

def main():
	
    cap = cv2.VideoCapture("/dev/video0")

    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame.")
            break

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def readim():

    pass

def cascade(cascade, im):

    pass

def center(tl_point, width):

    pass

def calc_pos(center):

    pass

def calc_adjust(x, y, size):

    pass

def pos_adjust(dx, dy, magnitude):

    pass

def sweep():

    pass

def greyscale():

    pass

def blur():

    pass

def canny():

    pass

if __name__ == "__main__":
    main()