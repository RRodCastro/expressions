from imutils.video import FileVideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
from time import sleep
from os import listdir

LANDMARKS = ['mouth', 'right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye']


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# vs = FileVideoStream(filePath).start()


# fileStream = True

time.sleep(1.0)


def getLandmarks(data, facial_landmarks, sex, truth, interview, frame):
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

    rigth_eyebrown_eye_distance = (dist.euclidean(
        right_eye[1], right_eyebrow[2]) + dist.euclidean(right_eye[2], right_eyebrow[3]))/2

    left_eyebrown_eye_distance = (dist.euclidean(
        left_eye[1], left_eyebrow[1]) + dist.euclidean(left_eye[2], left_eyebrow[2]))/2

    data.write(interview + ',' + frame + ',' + str(rigth_eyebrown_eye_distance * unit) + ',' +
               str(left_eyebrown_eye_distance * unit) + ',' +
               str(lips_width * unit) + ',' +
               str(lips_height * unit) + ',' +
               str(inner_mouth * unit) + ',' +
               str(sex) + ',' + str(truth) + '\n')


def detect_face_parts(detector, predictor, data, sex, truth, interview, frame):
    # frame = imutils.resize(frame, width=450)
    gray = cv2.imread("../videos-to-frames/beforeAnswer/%s/%s" %
                      (interview, frame))
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):

        face = dlib.get_face_chip(gray,  predictor(gray, rect))
        shape = predictor(face, detector(face, 0)[0])
        shape = face_utils.shape_to_np(shape)

        facial_landmarks = {}
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name in LANDMARKS:
                for (x, y) in shape[i:j]:
                    cv2.circle(face, (x, y), 1, (255, 69, 0), -1)

                facial_landmarks[name] = shape[i:j]
        cv2.imshow('frame', face)
        cv2.waitKey(0)
    # getLandmarks(data, facial_landmarks, sex, truth, interview, frame)


sex = 0
data = open("beforeAnswer2.csv", "a")

# H = [
#     'E002M', 'E002V', 'E003M', 'E003V', 'E006M', 'E006V',
#     'E010M', 'E010V', 'E013M', 'E013V', "E015V", "E015M",
#     "E036M", "E036V", "E037V", "E037M", 'E039M', 'E039V',
#     "E042V", "E042M", "E048M", "E048V", 'E052M', 'E052V',
#     'E063M', 'E063V', 'E064M',  "E066M", 'E066V', 'E068M', 'E068V']
# M = ['E014M', 'E014V', 'E016M', 'E016V', 'E018M', 'E018V',
#      'E026M', 'E026V', 'E027M', 'E027V', 'E030M', 'E030V',
#      'E031M', 'E031V', 'E033M', 'E033V', 'E034M', 'E034V',
#      'E038M', 'E038V', 'E045M', 'E045V', 'E046M', 'E046V',
#      'E054M', 'E054V', 'E060M', 'E060V', 'E062M', 'E062V',
#      'E065M', 'E065V', 'E067M', 'E067V']

M = ["E014M"]

# frames = listdir("../videos-to-frames/beforeAnswer")
# frames = list(map(lambda frame: frame.split(".")[0], frames))
# for i in frames:
#     if(i not in H):
#         M.append(i)
# print(M)
for interview in M:
    print("current interview ... " + interview)
    frames = listdir("../videos-to-frames/beforeAnswer/%s" % interview)
    frames = sorted(frames, key=lambda name: int(name.split("-")[0]))
    if ('V' in interview):
        truth = 1
    else:
        truth = 0
    for frame in frames:
        detect_face_parts(detector, predictor, data,
                          sex, truth, interview, frame)
data.close()
