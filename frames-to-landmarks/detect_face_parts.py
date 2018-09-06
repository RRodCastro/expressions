# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import sys
from time import sleep
import json

LANDMARKS = ['mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye']

fileName = sys.argv[1] if len(sys.argv) > 1 else "default"
geneder = sys.argv[2]
if ('V' in fileName):
    truth = 1
else:
    truth = 0


def getLandmarks(facial_landmarks, file):
    mouth = facial_landmarks[LANDMARKS[0]]
    right_eyebrow = facial_landmarks[LANDMARKS[1]]
    left_eyebrow = facial_landmarks[LANDMARKS[2]]
    right_eye = facial_landmarks[LANDMARKS[3]]
    left_eye = facial_landmarks[LANDMARKS[4]]

    center_rigth_eye = (
        right_eye[1][0] + ((right_eye[2][0] - right_eye[1][0]) / 2))

    center_left_eye = (left_eye[1][0] +
                       ((left_eye[2][0] - left_eye[1][0])/2))
    unit = (center_left_eye - center_rigth_eye) / 64

    lips_width = dist.euclidean(mouth[6], mouth[0])

    lips_height = (dist.euclidean(mouth[2], mouth[10]) +
                   dist.euclidean(mouth[4], mouth[8]))/2

    inner_mouth = (dist.euclidean(mouth[19], mouth[13])
                   + dist.euclidean(mouth[17], mouth[15]))/2

    left_eyebrown_eye_distance = abs(left_eye[0][1] - left_eyebrow[0][1])
    
    print(right_eye[1], right_eyebrow[2])

    sleep(20)

    # data.write(fileName + '$' + str(file) + '-' + str(rigth_eyebrown_eye_distance * unit) + '-' + str(left_eyebrown_eye_distance *
                                                                                                    #   unit) + '-' + str(lips_width * unit) + '-' + str(lips_height * unit) + '-' + str(inner_mouth * unit) + '-' + geneder + '-' + str(truth) + '\n')


dirPath = "../videos-to-frames/frames/%s" % fileName
dirPath = "./output"
if(os.path.exists(dirPath)):
    print(" exists ...")
    data = open("data.csv", "a")
else:
    print("Directory named %s doesnt exist" % dirPath)
    exit(0)
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

files = os.listdir(dirPath)
for file in range(len(files)):

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(dirPath + "/" + str(file) + ".jpg")
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        # loop over dict ('lamdnarkname', (x,y))
        clone = image.copy()
        facial_landmarks = {}
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

            if name in LANDMARKS:
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                facial_landmarks[name] = shape[i:j]

        # cv2.imwrite('./output/'+str(file)+".jpg", clone)
        getLandmarks(facial_landmarks, file)
data.close()
