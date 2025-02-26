import cv2
import numpy as np
import os

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set dataset path
data_path = "faces"
faces = []
labels = []
IMG_SIZE = (100, 100)  # Resize all images to a fixed size

if not os.path.exists(data_path):
    print(f"Error: Dataset folder '{data_path}' not found! Please collect images first.")
    exit()

for person_id in os.listdir(data_path):
    person_path = os.path.join(data_path, person_id)
    if not os.path.isdir(person_path):  # Skip files
        continue
    for image in os.listdir(person_path):
        img_path = os.path.join(person_path, image)

        # Read image
        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if face_img is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue

        # Resize the image
        face_img = cv2.resize(face_img, IMG_SIZE)

        faces.append(face_img)
        labels.append(int(person_id))

# Ensure at least one face was found
if len(faces) == 0:
    print("Error: No faces found in dataset. Make sure you have collected images first.")
    exit()

# Convert lists to NumPy arrays
faces = np.array(faces, dtype="uint8")
labels = np.array(labels)

# Train and save model
recognizer.train(faces, labels)
recognizer.save("trained_model.yml")

print("✅ Training complete! Model saved as trained_model.yml")
