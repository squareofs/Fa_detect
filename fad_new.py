import cv2
import os
subjects=["","sagar", "Shashank"]

print("Preparing image data ")


#training function
def training():
    faces=[]
    label=[]
    all_image1_list=os.listdir("images_data/sagar")
    print("Sagar all images ", all_image1_list)
    for image_name in all_image1_list:
        if image_name.startswith("."):
            continue;
        image_path = "images_data/sagar/" + image_name
        image = cv2.imread(image_path)
        cv2.imshow("Saving image data", cv2.resize(image, (400, 500)))
        cv2.waitKey(100)
        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            label.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()



    all_image2_list = os.listdir("images_data/shashank")
    print("Shashank all images ", all_image2_list)
    for image_name in all_image2_list:
        if image_name.startswith("."):
            continue;
        image_path = "images_data/shashank/" + image_name
        image = cv2.imread(image_path)
        cv2.imshow("Saving image data", cv2.resize(image, (400, 500)))
        cv2.waitKey(100)
        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            label.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, label

#detecting function
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files3/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


faces, label= training()
