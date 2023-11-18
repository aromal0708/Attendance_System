import cv2
import numpy as np
import os

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
FACE_IMAGES_DIR = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/Sample/user"
name = input("Enter your name or ID: ")
os.makedirs(os.path.join(FACE_IMAGES_DIR , name), exist_ok=True)

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    for x, y, w, h in faces:
        cropped_face = img[y : y + h, x : x + w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


        image_path = os.path.join(FACE_IMAGES_DIR, name, str(count) + ".jpg")
        cv2.imwrite(image_path, face)

        cv2.putText(
            face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Face Cropper", face)

    else:
        print("Face not found !!")
        pass

    if cv2.waitKey(1) == 13 or count == 150:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting samples completed *_*")

#training

data_path = os.path.join(FACE_IMAGES_DIR, name)
# Get the list of files in the data_path directory
onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = os.path.join(data_path, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if images is not None:
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    else:
        print(f"Image {file} not loaded successfully. Skipping...")

Labels = np.asarray(Labels, dtype=np.int32)

# Use the appropriate method based on the OpenCV version
if cv2.__version__.startswith('4'):
    model = cv2.face.LBPHFaceRecognizer_create()
else:
    model = cv2.face.createLBPHFaceRecognizer()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Congratulations model is TRAINED ... *_*...")