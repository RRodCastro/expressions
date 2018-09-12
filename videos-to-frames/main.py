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
    timesPath = "../../times/%s.csv" % path
    times = open(timesPath, 'r')
    arrTimes = []
    for i in times:
        arrTimes.append(i.split(","))
    # number of frames in videos
    frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # frames per second (30 something)
    FPS = stream.get(cv2.CAP_PROP_FPS)
    print(FPS)
    if skip == None:
        # Skip rate (grab every skip frame (every 1 frame))
        # skip = int(FPS)
        skip = int(FPS)
    index = 0
    second = 0
    searchTime = arrTimes.pop(0)
    while True:
        # skips some some of the frames, and read one
        if(second > int(searchTime[2].replace('\n', ''))):
            try:
                searchTime = arrTimes.pop(0)
            except:
                break

        (grabbed, frame) = stream.read()
        if not grabbed:
            return
        if(int(searchTime[1]) <= second <= int(searchTime[2].replace('\n', ''))):
            cv2.putText(frame, str(second), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite("./%s/%s.jpg" % ("./frames/"+fileName, index), frame)
        index += 1
        second = int(index/FPS)


fileName = sys.argv[1] if len(sys.argv) > 1 else ""
filePath = "../../videos/%s.mp4" % fileName


if (os.path.exists(filePath)):
    try:
        os.makedirs("./frames/%s" % fileName)
    except OSError as e:
        shutil.rmtree("./frames/%s" % fileName)
        os.makedirs(fileName)
        print("directory exists")
    videoStreamer(fileName)
else:
    print("File name %s doesnt exist" % fileName)
