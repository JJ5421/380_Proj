import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import math
import RPi.GPIO as GPIO

# Coefficients list
c1 = 1/(1920*1080) # Size movement multiplier
c2 = 1/1000 # Distance movemment multiplier
c3 = 200 # Minimum distance threshold
c4 = 1 # Converts 0-1 |----> servo degree change

# Servo positions for global storage
s_az = 50
s_el = 50
# Servo max position
s_max = 270
# Servo GPIO pins
s_p_az = 13
s_p_el = 19

#### EDIT #####
# Initiate GPIO
GPIO.setmode(GPIO.BCM) # Broadcom SOC numbers
pwm_az = GPIO.PWM(s_p_az, 50)
pwm_el = GPIO.PWM(s_p_el, 50)
# Set output pins
GPIO.output(s_p_az, True)
GPIO.output(s_p_el, True)
# Initial position
pwm_az.ChangeDutyCycle(30)
pwm_el.ChangeDutyCycle(30)

# Using built-in face cascade for now
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Must define viable camera (troubleshoot video0-4 if not working)
cap = cv2.VideoCapture("/dev/video0")
# Stops code in the case of failure
if not cap.isOpened():
    print("Error: Couldn't open the camera.")

# Find screen center point (x,y) necessary for location calculations
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Calculate the center coordinates
center_x = camera_width // 2
center_y = camera_height // 2
# Screen center tuple
screen_center = (center_x,center_y)

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

# Run cascade (specify both image and cascade)
def run_cascade(cascade, im):
    # Obtain grayscale
    im = grayscale(im)
    # Perform face detection
    objects = cascade.detectMultiScale(im, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    return objects

# Takes a set of cascade objects and finds the one nearest to the center of the screen
def find_closest(centers):

    # Indexes and storage values:
    # Counter
    i = 0
    # Index of nearest point
    small_index = 0
    # Smallest distance
    d_small = 10000

    # Check which center is closest to the screen absolute center
    for center in centers:
        # Distance from center to screen center
        dist = dist_2(center, screen_center)
        # If smaller than previous smallest distance, store value and indicator
        if dist <= d_small:
            d_small = dist
            small_index = i

    return small_index

# Finds the 2-D line distance between two points    
def dist_2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return (math.sqrt((dx*dx)+(dy*dy)))

# Finds the coordinate of the center of the object
def find_center(object_vals):
    # Collect values from cascade outputs
    x_tl = object_vals[0]
    y_tl = object_vals[1]
    width = object_vals[2]
    height = object_vals[3]
    # Calculate center point of object
    midx = x_tl - (width/2)
    midy = y_tl - (height/2)
    # Store center tuple
    center = (midx,midy)
    
    return center

# Calculates the magnitude and direction of adjustment necessary
def calc_adj(obj):
    # Center of trackable object
    center = find_center(obj)
    
    # Distance (pixels)
    dist = dist_2(center, screen_center)
    # Direction (angle)
    angle = math.arctan2((obj[0]-center_x, obj[1]-center_y))
    # Size (calculated as area)
    size = obj[2]*obj*[3]*c1

    # Outside bounding circle, so we need to calculate movement
    # Coefficients should be tuned to make this on the scale of (-1,1)
    if dist > c2:
        xc = math.cos(angle)*(dist*c2)*(size*c1)
        yc = math.sin(angle)*(dist*c2)*(size*c1)
        return (xc,yc)
    # Within central bounding circle, so we choose not to move
    else:
        return (0,0)

# This calculates the real necessary position adjustment to sevo positions necessary
def calc_servo_adj(xval, yval):
    # Convery adjustments to actionable servo PWM values
    az_adj = xval*c4
    el_adj = yval*c4
    # Calculate new azimuth and elevation servo values using the adjustments
    new_az = s_az + az_adj
    new_el = s_el + el_adj

    # If we were to attempt to move either servo beyond its operational limits, we need to stop at the max/min position
    if new_az > s_max:
        new_az = s_max
    elif new_az < 0:
        new_az = 0
    # If we were to attempt to move either servo beyond its operational limits, we need to stop at the max/min position
    if new_el > s_max:
        new_el = s_max
    elif new_el < 0:
        new_el = 0

    # Return new servo positions
    return(new_az, new_el)

# This pushes the position adjustments to the servos (actually moves them to track detected objects)
def pos_adjust(az, el):
    # Change the stored global servo values
    s_az = az
    s_el = el

    # Update the servo positions
    GPIO.pinwrite(s_p_az, s_az)
    GPIO.pinwrite(s_p_el, s_el)

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
def blur(frame):
    # Apply a gaussian blur to a frame
    blur = cv2.GaussianBlur(frame, (3,3))

    # Return blurred frame
    return blur

# Runs canny edge detection on a frame
def canny(frame):
    # Get the grescale of the image (helps edge detection)
    grayscale_image = grayscale(frame)
    # Get the gaussian-blurred version of the greyscale (pretty standard for edge detection, limits erroneous readings)
    blurred_image = blur(grayscale_image)

    # Run the canny edge detection algorithm with thresholds that produce a nice result on the OpenCV test image / logo
    canny_edges = cv2.Canny(blurred_image, 60, 100)

    return canny_edges

# Sets a servo angle
def set_angle(angle):

    pass

# Main function - we run this
def main():

    while True:
        frame = readim(cap)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

# Runs main function
if __name__ == "__main__":
    main()