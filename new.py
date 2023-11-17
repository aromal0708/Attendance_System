# Import the libraries
import face_recognition
import cv2
import os
import pickle
import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Define the constants
FACE_IMAGES_DIR = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/Sample/" # The directory that contains the face images
FACE_ENCODINGS_FILE = "face_encodings.pkl" # The file that contains the face encodings and the names
FACE_CLASSIFIER_FILE = "face_classifier.pkl" # The file that contains the trained classifier
ATTENDANCE_FILE = "attendance.csv" # The file that contains the attendance records
CAMERA_ID = 0 # The ID of the camera to use
FACE_DISTANCE_THRESHOLD = 0.6 # The threshold for face recognition
FACE_SCALE_FACTOR = 0.25 # The factor to resize the face image
FACE_DETECTION_MODEL = "cnn" # The model to use for face detection
KNN_N_NEIGHBORS = 5 # The number of neighbors for the KNN classifier
KNN_WEIGHTS = "distance" # The weights for the KNN classifier
KNN_ALGORITHM = "ball_tree" # The algorithm for the KNN classifier

# Define the functions
def sampling():
    # This function captures and saves the face images of the person
    # It asks the user to enter their name or ID and creates a subfolder with that name in the face images directory
    # It then opens the camera and captures 10 face images of the person with different poses and expressions
    # It saves the images in the subfolder with the name or ID of the person
    # It returns the name or ID of the person

    # Ask the user to enter their name or ID
    name = input("Enter your name or ID: ")

    # Create a subfolder with the name or ID of the person in the face images directory
    os.makedirs(os.path.join(FACE_IMAGES_DIR, name), exist_ok=True)

    # Open the camera
    cap = cv2.VideoCapture(CAMERA_ID)

    # Initialize the counter for the number of images
    count = 0

    # Loop until 10 images are captured
    while count < 10:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is valid
        if not ret:
            break

        # Convert the frame to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Detect the face locations in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)

        # Check if there is exactly one face in the frame
        if len(face_locations) == 1:
            # Get the face location
            top, right, bottom, left = face_locations[0]

            # Crop the face from the frame
            face = frame[top:bottom, left:right]

            # Save the face image in the subfolder with the name or ID of the person
            cv2.imwrite(os.path.join(FACE_IMAGES_DIR, name, f"{count}.jpg"), face)

            # Increment the counter
            count += 1

            # Draw a green rectangle around the face in the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Put the name or ID of the person and the number of images on the frame
            cv2.putText(frame, f"{name} - {count}/10", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow("Face Sampling", frame)

        # Wait for a key press
        key = cv2.waitKey(1)

        # Check if the key is ESC
        if key == 27:
            break

    # Release the camera
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()

    # Return the name or ID of the person
    return name

def train():
    # This function extracts the face encodings from the images and trains a classifier to recognize the faces
    # It iterates over the subfolders in the face images directory and loads the face images of each person
    # It then computes the face encodings for each image and stores them in a list along with the name or ID of the person
    # It then creates and trains a KNN classifier with the face encodings and the names as the features and the labels
    # It saves the face encodings and the names in a file and the classifier in another file
    # It returns the face encodings, the names, and the classifier

    # Initialize the lists for the face encodings and the names
    face_encodings = []
    names = []

    # Iterate over the subfolders in the face images directory
    for name in os.listdir(FACE_IMAGES_DIR):
        # Get the path of the subfolder
        subfolder = os.path.join(FACE_IMAGES_DIR, name)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder):
            # Iterate over the files in the subfolder
            for file in os.listdir(subfolder):
                # Get the path of the file
                file = os.path.join(subfolder, file)

                # Check if the file is an image
                if file.endswith(".jpg") or file.endswith(".png"):
                    # Load the image
                    image = face_recognition.load_image_file(file)

                    # Compute the face encoding for the image
                    encoding = face_recognition.face_encodings(image)[0]

                    # Append the encoding and the name to the lists
                    face_encodings.append(encoding)
                    names.append(name)

    # Create and train a KNN classifier with the face encodings and the names
    classifier = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS, algorithm=KNN_ALGORITHM)
    classifier.fit(face_encodings, names)

    # Save the face encodings and the names in a file
    with open(FACE_ENCODINGS_FILE, "wb") as f:
        pickle.dump((face_encodings, names), f)

    # Save the classifier in another file
    with open(FACE_CLASSIFIER_FILE, "wb") as f:
        pickle.dump(classifier, f)

    # Return the face encodings, the names, and the classifier
    return face_encodings, names, classifier

def recognise():
    # This function recognizes the face and marks the attendance of the person who shows their face to the camera
    # It loads the face encodings and the names from a file and the classifier from another file
    # It then opens the camera and reads a frame from it
    # It then resizes the frame and detects the face locations in it
    # It then computes the face encodings for each face and predicts the name or ID of the person using the classifier
    # It then records the date and time of the attendance and appends them to a file or a database
    # It also displays the result and the attendance sheet on the screen
    # It returns the name or ID of the person and the date and time of the attendance

    # Load the face encodings and the names from a file
    with open(FACE_ENCODINGS_FILE, "rb") as f:
        face_encodings, names = pickle.load(f)

    # Load the classifier from another file
    with open(FACE_CLASSIFIER_FILE, "rb") as f:
        classifier = pickle.load(f)

    # Open the camera
    cap = cv2.VideoCapture(CAMERA_ID)

    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame