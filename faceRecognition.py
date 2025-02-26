import cv2
import numpy as np
import os

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the trained face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')  # Make sure you have a trained model

# Labels for people (Ensure you map them correctly)
labels = {0: "Greycin", 1: "Brian", 2: "Noah", 3: "Jason", 4:"Ellie", 5: "Greycin", 6: "Matthew"}

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Region of interest (face)

        # Recognize face
        label, confidence = recognizer.predict(face_roi)

        # Draw rectangle and label
        name = labels.get(label, "Unknown")
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, f"{name} ({int(confidence)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", img)

    # Exit on ESC
    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
