# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
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
    print(file)
    mouth = facial_landmarks[LANDMARKS[0]]
    right_eyebrow = facial_landmarks[LANDMARKS[1]]
    left_eyebrow = facial_landmarks[LANDMARKS[2]]
    right_eye = facial_landmarks[LANDMARKS[3]]
    left_eye = facial_landmarks[LANDMARKS[4]]

    # mouth = np.array(ast.literal_eval(mouth))
    # right_eyebrow = np.array(ast.literal_eval(right_eyebrow))
    # left_eyebrow = np.array(ast.literal_eval(left_eyebrow))
    # right_eye = np.array(ast.literal_eval(right_eye))
    # left_eye = np.array(ast.literal_eval(left_eye))
    center_rigth_eye = (
        right_eye[1][0] + ((right_eye[2][0] - right_eye[1][0]) / 2))

    center_left_eye = (left_eye[1][0] +
                       ((left_eye[2][0] - left_eye[1][0])/2))
    unit = (center_left_eye - center_rigth_eye) / 64

    lips_width = abs(mouth[6][0] - mouth[0][0])
    lips_height = (abs(mouth[2][1] - mouth[10][1]) +
                   abs(mouth[4][1] - mouth[8][1]))/2

    inner_mouth = (abs(mouth[19][1] - mouth[13][1]) + abs(mouth[18]
                                                          [1] - mouth[14][1]) + abs(mouth[17][1] - mouth[15][1]))/3

    rigth_eyebrown_eye_distance = abs(
        right_eye[3][1] - right_eyebrow[4][1])
    left_eyebrown_eye_distance = abs(left_eye[0][1] - left_eyebrow[0][1])

    data.write(fileName + '$' + str(file) + '-' + str(rigth_eyebrown_eye_distance * unit) + '-' + str(left_eyebrown_eye_distance *
                                                                                                      unit) + '-' + str(lips_width * unit) + '-' + str(lips_height * unit) + '-' + str(inner_mouth * unit) + '-' + geneder + '-' + str(truth) + '\n')


dirPath = "../videos-to-frames/%s" % fileName

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
        facial_landmarks = {}
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

            if name in LANDMARKS:
                facial_landmarks[name] = shape[i:j]

        getLandmarks(facial_landmarks, file)
data.close()
