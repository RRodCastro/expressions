from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
from time import sleep

LANDMARKS = ['mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye']


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# vs = FileVideoStream(filePath).start()


# fileStream = True

time.sleep(1.0)

def getLandmarks(data, facial_landmarks, sex, truth):
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

    data.write(str(rigth_eyebrown_eye_distance * unit) + '-' + str(left_eyebrown_eye_distance *
                                                                                                      unit) + '-' + str(lips_width * unit) + '-' + str(lips_height * unit) + '-' + str(inner_mouth * unit) + '-' + sex + '-' + str(truth) + '\n')

def detect_face_parts(detector, predictor, filePath, sex):
    if ('V' in fileName):
        truth = 1
    else:
        truth = 0
    data = open("data.csv", "a")

    stream = cv2.VideoCapture(filePath)
    while True:
    # if fileStream and not vs.more():
    #     break 
    #skip 5 frames
        for i in range(5):
            stream.grab()
            (grabbed, frame) = stream.read()
        if not grabbed:
            return

        # frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            facial_landmarks = {}
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name in LANDMARKS:
                    # for (x, y) in shape[i:j]:
                    facial_landmarks[name] = shape[i:j]
                        # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            getLandmarks(data, facial_landmarks, sex, truth)
            # cv2.imshow("MP", frame)
            # key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break
    
    data.close()
    # cv2.destroyAllWindows()

fileName = sys.argv[1] if len(sys.argv) > 1 else ""
sex = sys.argv[2]

filePath = "../../videos/%s.mp4" % fileName

detect_face_parts(detector, predictor, filePath, sex)
