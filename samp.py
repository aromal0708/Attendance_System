import cv2
import numpy as np
import os
import pandas as pd

# Define the FACE_IMAGES_DIR as the directory where the user folders are stored
FACE_IMAGES_DIR = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/Sample/user"
# Define the excel file name and the sheet name
user_names = os.listdir(FACE_IMAGES_DIR)

excel_file = "attendance.xlsx"
sheet_name = "Sheet1"
# Define the path of the excel file
excel_path = os.path.join("C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/", excel_file)
# Define the model file name and the path
model_file = "face_model.yml"
model_path = os.path.join("C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/", model_file)

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

# Ask the user to enter their name or ID
name = input("Enter your name or ID: ")
# Create a folder with the name of the user if it does not exist
os.makedirs(os.path.join(FACE_IMAGES_DIR , name), exist_ok=True)

# Check if the model file exists
if os.path.isfile(model_path):
    # Load the model from the file
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_path)
    print("Model loaded successfully.")
else:
    # Train the model from scratch
    Training_Data, Labels = [], []
    # Get the list of user names from the folder names
    user_names = os.listdir(FACE_IMAGES_DIR)
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
    # Save the model to the file
    model.write(model_path)
    print("Model trained and saved successfully.")

# Load the existing dataframe from the excel file
if os.path.isfile(excel_path):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Name")
else:
    # Create a pandas dataframe to store the user names and the attendance status
    df = pd.DataFrame(columns=["Name", "Attendance", "Time"])
    # Set the index of the dataframe as the user names
    df.set_index("Name", inplace=True)

# Loop through each user name and initialize the attendance status as "Absent" if not already present
for user_name in user_names:
    if user_name not in df.index or pd.isnull(df.loc[user_name, "Attendance"]):
        df.loc[user_name] = ["Absent", None]

# Save the dataframe to the excel file
df.to_excel(excel_path, sheet_name=sheet_name)

cap = cv2.VideoCapture(0)
# Initialize the sample count
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Use os.path.join instead of + to concatenate the paths
        image_path = os.path.join(FACE_IMAGES_DIR, name, str(count) + ".jpg")
        cv2.imwrite(image_path, face)

        cv2.putText(
            face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Face Cropper", face)

    else:
        print("Collecting Samples")
        pass

    if cv2.waitKey(1) == 13 or count == 100: # Assuming you want to break the loop if 'Enter' is pressed or 100 samples are collected
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
