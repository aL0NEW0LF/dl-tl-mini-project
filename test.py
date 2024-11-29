"""import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "D:\\Projects\\dl-tl-mini-project\\model.h5",
    "D:\\Projects\\dl-tl-mini-project\\model/labels.txt",
)
offset = 20
imgSize = 300
counter = 0

labels = [
    "ع",
    "ال",
    "أ",
    "ب",
    "د",
    "ظ",
    "ض",
    "ف",
    "ق",
    "غ",
    "ه",
    "ح",
    "ج",
    "ك",
    "خ",
    "لا",
    "ل",
    "م",
    "ن",
    "ر",
    "ص",
    "س",
    "ش",
    "ط",
    "ت",
    "ث",
    "ذ",
    "ة",
    "و",
    "ئ",
    "ي",
    "ز",
]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap : wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap : hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset - 70),
            (x - offset + 400, y - offset + 60 - 50),
            (0, 255, 0),
            cv2.FILLED,
        )

        cv2.putText(
            imgOutput,
            labels[index],
            (x, y - 30),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (0, 0, 0),
            2,
        )
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (0, 255, 0),
            4,
        )

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
from classifier import Classifier
import numpy as np
import math
import os
import time

classifier = Classifier(
    "D:\\Projects\\dl-tl-mini-project\\model.h5",
    "D:\\Projects\\dl-tl-mini-project\\model/labels.txt",
)
offset = 40
imgSize = 64
counter = 0

labels = os.listdir("assets\\data\\ArASL_Database_54K")

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(
    staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5
)

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()

    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        # lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        bbox1 = hand1[
            "bbox"
        ]  # Bounding box around the first hand (x,y,w,h coordinates)
        x, y, w, h = bbox1

        if x >= 0 and y >= 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 128

            imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]
            imgCropShape = imgCrop.shape

            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    newImg = img[
                        y - offset : y + h + offset,
                        x - offset - wGap : x + w + offset + wGap,
                    ]
                    imgWhite = newImg
                    """imgWhite[:, wGap : wCal + wGap] = imgResize"""
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    newImg = img[
                        y - offset - hGap : y + h + offset + hGap,
                        x - offset : x + w + offset,
                    ]
                    imgWhite = newImg
                    """imgWhite[hGap : hCal + hGap, :] = imgResize"""

                imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                imgWhite = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
                print(imgWhite.shape)

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index, labels[index])
                """cv2.namedWindow("ImageB", cv2.WINDOW_NORMAL)
                cv2.imshow("ImageB", imgWhite)
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("a"):
                        break
                cv2.destroyWindow("ImageB")"""

            """bounding_box = img[y : y + h, x : x + w]
            gray_img = cv2.cvtColor(bounding_box, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (imgSize, imgSize))
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            gray_img = np.expand_dims(gray_img, axis=0)
            prediction, index = classifier.getPrediction(gray_img, draw=False)
            print(prediction, labels[index])"""

    # Display the image with the detected hands
    cv2.imshow("Image", img)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
