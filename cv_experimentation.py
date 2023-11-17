import cv2

import urllib.request

# Download a pre-trained Haar Cascade XML file for face detection
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, 'cascade.xml')


def view_webcam():
    # Create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier('cascade.xml')

    while True:
        # Read frames from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        # Draw bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the captured frame with bounding boxes
        cv2.imshow('Webcam', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to view webcam footage with face detection
view_webcam()