import cv2
import numpy as np
import os
import pandas as pd
import datetime

# Define the FACE_IMAGES_DIR as the directory where the user folders are stored
FACE_IMAGES_DIR = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/Sample/user"
# Get the list of user names from the folder names
user_names = os.listdir(FACE_IMAGES_DIR)

Training_Data, Labels = [], []

# Loop through each user folder and load the images and labels
for i, user_name in enumerate(user_names):
    # Define the data_path as the user folder
    data_path = os.path.join(FACE_IMAGES_DIR, user_name)
    # Get the list of files in the data_path directory
    onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    # Loop through each file and load the image and the label
    for file in onlyfiles:
        image_path = os.path.join(data_path, file)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Check if the image is loaded successfully
        if images is not None:
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)
        else:
            print(f"Image {file} not loaded successfully. Skipping...")

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Congratulations model is TRAINED ... *_*...")

face_classifier = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    for x, y, w, h in faces:
        cropped_face = img[y : y + h, x : x + w]

    return cropped_face


def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y : y + h, x : x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi


# Define the excel file name and the sheet name
excel_file = "attendance.xlsx"
sheet_name = "Sheet1"
# Define the path of the excel file
excel_path = os.path.join("C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/", excel_file)

# Load the existing dataframe from the excel file
if os.path.isfile(excel_path):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Name")
else:
    # Create a pandas dataframe to store the user names and the attendance status
    df = pd.DataFrame(columns=["Name", "Attendance", "Time"])
    # Set the index of the dataframe as the user names
    df.set_index("Name", inplace=True)
    # Loop through each user name and initialize the attendance status as "Absent"
    for user_name in user_names:
        df.loc[user_name] = ["Absent", None]
    # Save the dataframe to the excel file
    df.to_excel(excel_path, sheet_name=sheet_name)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            Confidence = int(100 * (1 - (result[1]) / 300))
            display_string = f"{Confidence}% confidence it is user"
        cv2.putText(
            image,
            display_string,
            (100, 120),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (250, 120, 255),
            2,
        )

        if Confidence > 65:
            user_name = user_names[result[0]]
            cv2.putText(
                image,
                f"HELLO {user_name}",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Cropper", image)
            # Load the existing dataframe from the excel file
            if os.path.isfile(excel_path):
                df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Name")
            # Check if the user is already marked as present
            if df.loc[user_name, "Attendance"] != "Present":
                # Mark the user as present and save the current time
                df.loc[user_name] = ["Present", datetime.datetime.now()]
                # Save the dataframe to the excel file
                df.to_excel(excel_path, sheet_name=sheet_name)

        else:
            cv2.putText(
                image,
                "UNKNOWN",
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Face Cropper", image)

    except:
        cv2.putText(
            image,
            "NO FACE FOUND",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Face Cropper", image)
        pass

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
