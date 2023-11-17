# # training model

# import cv2  # open cv library
# import numpy as np  # for mathematical calculation
# from os import listdir  # class of os module // when fetching data
# from os.path import isfile, join

# data_path = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/Sample/"
# onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Training_Data, Labels = [], []

# for i, files in enumerate(onlyfiles):
#     image_path = data_path + onlyfiles[i]
#     images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     Training_Data.append(np.asarray(images, dtype=np.uint8))
#     Labels.append(i)

# Labels = np.asarray(Labels, dtype=np.int32)

# if cv2.__version__.startswith('4'):
#     model = cv2.face.LBPHFaceRecognizer_create()
# else:
#     model = cv2.createLBPHFaceRecognizer()


# model.train(np.asarray(Training_Data), np.asarray(Labels))

# print("Congratulations model is TRAINED ... *_*...")

# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/AROMAL SUNIL/OneDrive/Desktop/Env/Project/Sample/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(data_path, file)
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
