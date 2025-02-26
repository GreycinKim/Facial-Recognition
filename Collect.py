import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

person_id = input("Enter a numeric ID for this person: ")
save_path = f"faces/{person_id}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0
while count < 50:  # Collect 50 images
    _, img = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        cv2.imwrite(f"{save_path}/{count}.jpg", face_roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        count += 1

    cv2.imshow("Collecting Faces", img)
    if cv2.waitKey(10) == 27 or count >= 50:
        break

webcam.release()
cv2.destroyAllWindows()
