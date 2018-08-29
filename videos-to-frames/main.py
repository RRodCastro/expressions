import cv2
import imutils
import time
import sys
import argparse
import base64
import os
import simplejson as json
from PIL import Image
from time import sleep
import shutil


def videoStreamer(path, width=600, height=600, skip=None):
    # capture the video
    filePath = "../../videos/%s.mp4" % path
    stream = cv2.VideoCapture(filePath)
    # number of frames in videos
    frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # frames per second (30 something)
    FPS = stream.get(cv2.CAP_PROP_FPS)
    if skip == None:
        # Skip rate (grab every skip frame (every 1 frame))
        # skip = int(FPS)
        skip = int(FPS)
    index = 0
    while True:
        # skips some some of the frames, and read one
        for i in range(5):
            stream.grab()
        (grabbed, frame) = stream.read()
        if not grabbed:
            return

        cv2.imwrite("./%s/%s.jpg" % (fileName, index), frame)

        index += 1


fileName = sys.argv[1] if len(sys.argv) > 1 else ""
filePath = "../../videos/%s.mp4" % fileName
print(filePath)
if (os.path.exists(filePath)):
    try:
        os.makedirs(fileName)
    except OSError as e:
        shutil.rmtree("./%s" % fileName)
        os.makedirs(fileName)
        print("directory exists")
    videoStreamer(fileName)
else:
    print("File name %s doesnt exist" % fileName)
