import cv2
import os
import numpy as np

subjects=["","Sagar", "Shashank"]
print("Preparing image data ")

#detecting function
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files3/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]



faces=[]
labels=[]
all_image1_list=os.listdir("images_data/p1")
print("Person 1: all images :", all_image1_list," \n Total images ",len(all_image1_list))
label=1
for image_name in all_image1_list:
    if image_name.startswith("."):
        continue;
    image_path="images_data/p1/" + image_name
    image=cv2.imread(image_path)
    cv2.imshow("Saving image data", cv2.resize(image, (400, 500)))
    cv2.waitKey(100)
    face, rect = detect_face(image)
    if face is not None:
        faces.append(face)
        labels.append(label)

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()



all_image2_list = os.listdir("images_data/p2")
print("Person 2: All images: ", all_image2_list," \n Total images ",len(all_image2_list))
label=2
for image_name in all_image2_list:
    if image_name.startswith("."):
        continue;
    image_path="images_data/p2/"+image_name
    image=cv2.imread(image_path)
    cv2.imshow("Saving image data",cv2.resize(image,(400,500)))
    cv2.waitKey(100)
    face, rect=detect_face(image)
    if face is not None:
        faces.append(face)
        labels.append(label)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()




print("\n \n All the labels Assigned")
print(labels)

print("Data prepared")
print("Total faces: ",len(faces))
print("Total labels: ",len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img

print("Predicting images...")
test_img1 = cv2.imread("input_image/1.jpg")
test_img2 = cv2.imread("input_image/2.jpg")
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
