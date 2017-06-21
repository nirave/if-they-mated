import numpy as np
import cv2
from os import path

#Note, this uses assignment 6's blending
import blending_helper as a6

IMG_FOLDER = "images"

def change_part(orig_img, orig_img2, alter_img1, type="eyes"):

    def get_parts(img, mask, type="eyes"):
        # Face/eye recognition copied from: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

        # Setup the system
        # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

        # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        # nose cascade
        #https://github.com/AlexeyAB/OpenCV-detection-models/blob/master/haarcascades/haarcascade_mcs_nose.xml
        nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

        # mouth cascade
        # https://github.com/AlexeyAB/OpenCV-detection-models/blob/master/haarcascades/haarcascade_mcs_mouth.xml
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

        # find the face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = None

        # Find the eyes
        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            if eyes.shape[0] < 2:
                print "Two eyes not found, aborting"
                exit(0)

            eye_one_x1 = eyes[0][0] + x
            eye_one_y1 = eyes[0][1] + y
            eye_one_x2 = eyes[0][2] + eye_one_x1
            eye_one_y2 = eyes[0][3] + eye_one_y1

            eye_two_x1 = eyes[1][0] + x
            eye_two_y1 = eyes[1][1] + y
            eye_two_x2 = eyes[1][2] + eye_two_x1
            eye_two_y2 = eyes[1][3] + eye_two_y1

            #Nose stuff
            nose = nose_cascade.detectMultiScale(roi_gray)

            i = 0
            nose_x1 = nose[i][0] + x
            nose_y1 = nose[i][1] + y
            nose_x2 = nose[i][2] + nose_x1
            nose_y2 = nose[i][3] + nose_y1

            # mouth
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            #print mouth

            #The mouth should be lower than the eyes
            i = 0
            for i in range(0, len(mouth)):
                if (mouth[i][3] + mouth[i][1] + x) > eye_one_y1 and(mouth[i][3] + mouth[i][1] + x)  > eye_one_y2:
                    break

            mouth_x1 = mouth[i][0] + x
            mouth_y1 = mouth[i][1] + y
            mouth_x2 = mouth[i][2] + mouth_x1
            mouth_y2 = mouth[i][3] + mouth_y1

            parts = (eye_one_x1, eye_one_y1, eye_one_x2, eye_one_y2, eye_two_x1, eye_two_y1, eye_two_x2, eye_two_y2,
                     nose_x1, nose_y1, nose_x2, nose_y2, mouth_x1, mouth_y1, mouth_x2, mouth_y2)

            return parts

    img = orig_img.copy()
    img2 = orig_img2.copy()

    mask = np.ones((img.shape[0], img.shape[1])) * 255
    (eye_one_x1, eye_one_y1, eye_one_x2, eye_one_y2, eye_two_x1, eye_two_y1, eye_two_x2, eye_two_y2, nose_x1, nose_y1,
     nose_x2, nose_y2, mouth_x1, mouth_y1, mouth_x2, mouth_y2) = get_parts(img, mask, type)

    (eye_one_x1_2, eye_one_y1_2, eye_one_x2_2, eye_one_y2_2, eye_two_x1_2, eye_two_y1_2, eye_two_x2_2, eye_two_y2_2,
     nose_x1_2, nose_y1_2, nose_x2_2, nose_y2_2, mouth_x1_2, mouth_y1_2, mouth_x2_2, mouth_y2_2) = get_parts(img2, mask, type)

    if "eye" in type:
        #for diagnostic purposes
        #cv2.rectangle(img, (eye_one_x1, eye_one_y1), (eye_one_x2, eye_one_y2), (0, 255, 0), 2)
        #cv2.rectangle(img, (eye_two_x1, eye_two_y1), (eye_two_x2, eye_two_y2), (0, 255, 0), 2)

        #Create the mask
        cv2.rectangle(mask, (eye_one_x1 - 10, eye_one_y1 - 10), (eye_one_x2 + 10, eye_one_y2 + 10),  (192, 192, 192), -1)
        cv2.rectangle(mask, (eye_one_x1 - 5, eye_one_y1 - 5), (eye_one_x2 + 5, eye_one_y2 + 5), (128, 128, 128), -1)
        cv2.rectangle(mask, (eye_one_x1 - 2, eye_one_y1 - 2), (eye_one_x2 + 2, eye_one_y2 + 2), (64, 64, 64), -1)
        cv2.rectangle(mask, (eye_one_x1, eye_one_y1), (eye_one_x2, eye_one_y2), (0, 0, 0), -1)
        cv2.rectangle(mask, (eye_two_x1 - 10, eye_two_y1 - 10), (eye_two_x2 + 10, eye_two_y2 + 10), (192, 192, 192), -1)
        cv2.rectangle(mask, (eye_two_x1 - 5, eye_two_y1 - 5), (eye_two_x2 + 5, eye_two_y2 + 5), (128, 128, 128), -1)
        cv2.rectangle(mask, (eye_two_x1 - 2, eye_two_y1 - 2), (eye_two_x2 + 2, eye_two_y2 + 2), (64, 64, 64), -1)
        cv2.rectangle(mask, (eye_two_x1, eye_two_y1), (eye_two_x2, eye_two_y2), (0, 0, 0), -1)

        midX1 = (eye_one_x1 + eye_one_x2) / 2
        midY1 = (eye_one_y1 + eye_one_y2) / 2

        midX2 = (eye_two_x1 + eye_two_x2) / 2
        midY2 = (eye_two_y1 + eye_two_y2) / 2

        midX1_2 = (eye_one_x1_2 + eye_one_x2_2) / 2
        midY1_2 = (eye_one_y1_2 + eye_one_y2_2) / 2
        xRadius1 = (eye_one_x2_2 - eye_one_x1_2) / 2 + 5
        yRadius1 = (eye_one_y2_2 - eye_one_y1_2) / 2 + 5

        midX2_2 = (eye_two_x1_2 + eye_two_x2_2) / 2
        midY2_2 = (eye_two_y1_2 + eye_two_y2_2) / 2
        xRadius2 = (eye_two_x2_2 - eye_two_x1_2) / 2 + 5
        yRadius2 = (eye_two_y2_2 - eye_two_y1_2) / 2 + 5

        #Copy the eyes to the correct place
        img2[midY1 - xRadius1:midY1 + xRadius1, midX1 - xRadius1:midX1 + xRadius1] = img2[midY1_2 - yRadius1:midY1_2 + xRadius1, midX1_2 - xRadius1:midX1_2 + xRadius1]
        img2[midY2 - xRadius2:midY2 + xRadius2, midX2 - xRadius2:midX2 + xRadius2] = img2[midY2_2 - yRadius2:midY2_2 + xRadius2, midX2_2 - xRadius2:midX2_2 + xRadius2]


    if "nose" in type:
        #copy the nose
        nose_midX1 = (nose_x1 + nose_x2) / 2
        nose_midY1 = (nose_y1 + nose_y2) / 2

        nose_midX1_2 = (nose_x1_2 + nose_x2_2) / 2
        nose_midY1_2 = (nose_y1_2 + nose_y2_2) / 2

        nose_xRadius1 = (nose_x2_2 - nose_x1_2) / 2 + 5
        nose_yRadius1 = (nose_y2_2 - nose_y1_2) / 2 + 5

        cv2.rectangle(mask, (nose_x1 - 5, nose_y1 - 5), (nose_x2 + 5, nose_y2 + 5), (64, 64, 64), -1)
        cv2.rectangle(mask, (nose_x1 - 2, nose_y1 - 2), (nose_x2 + 2, nose_y2 + 2), (128, 128, 128), -1)
        cv2.rectangle(mask, (nose_x1, nose_y1), (nose_x2, nose_y2), (0, 0, 0), -1)

        img2[nose_midY1 - nose_yRadius1:nose_midY1 + nose_yRadius1, nose_midX1 - nose_xRadius1:nose_midX1 + nose_xRadius1] \
            = img2[nose_midY1_2 - nose_yRadius1:nose_midY1_2 + nose_yRadius1,
              nose_midX1_2 - nose_xRadius1:nose_midX1_2 + nose_xRadius1 ]

    if "mouth" in type:
        # copy the mouth
        mouth_midX1 = (mouth_x1 + mouth_x2) / 2
        mouth_midY1 = (mouth_y1 + mouth_y2) / 2

        mouth_midX1_2 = (mouth_x1_2 + mouth_x2_2) / 2
        mouth_midY1_2 = (mouth_y1_2 + mouth_y2_2) / 2

        mouth_xRadius1 = (mouth_x2_2 - mouth_x1_2) / 2 + 5
        mouth_yRadius1 = (mouth_y2_2 - mouth_y1_2) / 2 + 5

        cv2.rectangle(mask, (mouth_x1 - 5, mouth_y1 - 5), (mouth_x2 + 5, mouth_y2 + 5), (128, 128, 128), -1)
        cv2.rectangle(mask, (mouth_x1 - 2, mouth_y1 - 2), (mouth_x2 + 2, mouth_y2 + 2), (64, 64, 64), -1)
        cv2.rectangle(mask, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2), (0, 0, 0), -1)

        img2[mouth_midY1 - mouth_yRadius1:mouth_midY1 + mouth_yRadius1,
        mouth_midX1 - mouth_xRadius1:mouth_midX1 + mouth_xRadius1] \
            = img2[mouth_midY1_2 - mouth_yRadius1:mouth_midY1_2 + mouth_yRadius1,
              mouth_midX1_2 - mouth_xRadius1:mouth_midX1_2 + mouth_xRadius1]

    # For debugging purposes only
    #cv2.imwrite("face.png", img)
    #cv2.imwrite("face2.png", img2)
    cv2.imwrite("face_mask.png", mask)

    first = alter_img1
    second = img2

    mask = cv2.imread("face_mask.png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(dtype=np.float64) / 255

    return a6.blender(img2, alter_img1, mask)



def if_they_mated(img1, img2, eyes=True, nose=True, mouth=True):
    final = img1

    if eyes:
        final = change_part(final, img2, img1, "eyes")
    if nose:
        final = change_part(img1, img2, final, "nose")
    if mouth:
        final = change_part(img1, img2, final, "mouth")
    return final


#Selected sample of myself and my daugher
anura = cv2.imread(path.join(IMG_FOLDER, "nirave.jpg"))
nirave = cv2.imread(path.join(IMG_FOLDER, "anura.jpg"))

cv2.imwrite("anura_nirave_all.png", if_they_mated(anura, nirave))
cv2.imwrite("anura_nirave_eyes.png", if_they_mated(anura, nirave, eyes=True, nose=False, mouth=False))
cv2.imwrite("anura_nirave_nose.png", if_they_mated(anura, nirave, eyes=False, nose=True, mouth=False))
cv2.imwrite("anura_nirave_mouth.png", if_they_mated(anura, nirave, eyes=False, nose=False, mouth=True))
cv2.imwrite("anura_nirave_eyes_nose.png", if_they_mated(anura, nirave, eyes=True, nose=True, mouth=False))
cv2.imwrite("anura_nirave_eyes_mouth.png", if_they_mated(anura, nirave, eyes=True, nose=False, mouth=True))
cv2.imwrite("anura_nirave_mouth_nose.png", if_they_mated(anura, nirave, eyes=False, nose=True, mouth=True))

#Selected sample of Kim Jong Un and Abe
kim = cv2.imread(path.join(IMG_FOLDER, "kim.jpg"))
abe = cv2.imread(path.join(IMG_FOLDER, "abe.jpg"))

cv2.imwrite("kim_abe_all.png", if_they_mated(kim, abe))
cv2.imwrite("kim_abe_eyes.png", if_they_mated(kim, abe, eyes=True, nose=False, mouth=False))
cv2.imwrite("kim_abe_nose.png", if_they_mated(kim, abe, eyes=False, nose=True, mouth=False))
cv2.imwrite("kim_abe_mouth.png", if_they_mated(kim, abe, eyes=False, nose=False, mouth=True))
cv2.imwrite("kim_abe_eyes_nose.png", if_they_mated(kim, abe, eyes=True, nose=True, mouth=False))
cv2.imwrite("kim_abe_eyes_mouth.png", if_they_mated(kim, abe, eyes=True, nose=False, mouth=True))
cv2.imwrite("kim_abe_mouth_nose.png", if_they_mated(kim, abe, eyes=False, nose=True, mouth=True))

#Selected sample of Hillary and Putin
hillary = cv2.imread(path.join(IMG_FOLDER, "hillary.jpg"))
putin = cv2.imread(path.join(IMG_FOLDER, "putin.jpg"))

cv2.imwrite("hillary_putin_all.png", if_they_mated(hillary, putin))
cv2.imwrite("hillary_putin_eyes.png", if_they_mated(hillary, putin, eyes=True, nose=False, mouth=False))
cv2.imwrite("hillary_putin_nose.png", if_they_mated(hillary, putin, eyes=False, nose=True, mouth=False))
cv2.imwrite("hillary_putin_mouth.png", if_they_mated(hillary, putin, eyes=False, nose=False, mouth=True))
cv2.imwrite("hillary_putin_eyes_nose.png", if_they_mated(hillary, putin, eyes=True, nose=True, mouth=False))
cv2.imwrite("hillary_putin_eyes_mouth.png", if_they_mated(hillary, putin, eyes=True, nose=False, mouth=True))
cv2.imwrite("hillary_putin_mouth_nose.png", if_they_mated(hillary, putin, eyes=False, nose=True, mouth=True))

#Selected sample of Trump and Putin
trump = cv2.imread(path.join(IMG_FOLDER, "trump.jpg"))
putin = cv2.imread(path.join(IMG_FOLDER, "putin.jpg"))

cv2.imwrite("trump_putin_all.png", if_they_mated(trump, putin))
cv2.imwrite("trump_putin_eyes.png", if_they_mated(trump, putin, eyes=True, nose=False, mouth=False))
cv2.imwrite("trump_putin_nose.png", if_they_mated(trump, putin, eyes=False, nose=True, mouth=False))
cv2.imwrite("trump_putin_mouth.png", if_they_mated(trump, putin, eyes=False, nose=False, mouth=True))
cv2.imwrite("trump_putin_eyes_nose.png", if_they_mated(trump, putin, eyes=True, nose=True, mouth=False))
cv2.imwrite("trump_putin_eyes_mouth.png", if_they_mated(trump, putin, eyes=True, nose=False, mouth=True))
cv2.imwrite("trump_putin_mouth_nose.png", if_they_mated(trump, putin, eyes=False, nose=True, mouth=True))


#Selected sample of Kim Jong Un and Abe
kim = cv2.imread(path.join(IMG_FOLDER, "kim.jpg"))
hillary = cv2.imread(path.join(IMG_FOLDER, "hillary.jpg"))

cv2.imwrite("kim_hillary_all.png", if_they_mated(kim, hillary))
cv2.imwrite("kim_hillary_eyes.png", if_they_mated(kim, hillary, eyes=True, nose=False, mouth=False))
cv2.imwrite("kim_hillary_nose.png", if_they_mated(kim, hillary, eyes=False, nose=True, mouth=False))
cv2.imwrite("kim_hillary_mouth.png", if_they_mated(kim, hillary, eyes=False, nose=False, mouth=True))
cv2.imwrite("kim_hillary_eyes_nose.png", if_they_mated(kim, hillary, eyes=True, nose=True, mouth=False))
cv2.imwrite("kim_hillary_eyes_mouth.png", if_they_mated(kim, hillary, eyes=True, nose=False, mouth=True))
cv2.imwrite("kim_hillary_mouth_nose.png", if_they_mated(kim, hillary, eyes=False, nose=True, mouth=True))


