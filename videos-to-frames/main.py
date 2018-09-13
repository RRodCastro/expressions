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
    # frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # frames per second (30 something)
    FPS = stream.get(cv2.CAP_PROP_FPS)
    if skip == None:
        # Skip rate (grab every skip frame (every 1 frame))
        # skip = int(FPS)
        skip = int(FPS)
    index = 0
    second = 0
    numberQuest = 1
    searchTime = arrTimes.pop(0)
    while True:
        # skips some some of the frames, and read one
        if(second > int(searchTime[2].replace('\n', ''))):
            try:
                searchTime = arrTimes.pop(0)
                numberQuest += 1
            except:
                break

        (grabbed, frame) = stream.read()
        if not grabbed:
            return
        if((index % 3 == 0) & (int(searchTime[1]) <= second <= int(searchTime[2].replace('\n', '')))):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            namedFrame = str(index) + "-" + \
                str(numberQuest) + "-" + str(second)
            cv2.imwrite("./%s/%s.jpg" %
                        ("./frames/"+fileName, namedFrame), gray)
        index += 1
        second = int(index/FPS)


files = ['E015M', 'E015V', 'E016M', 'E016V', 'E018M', 'E018V', 'E027M', 'E027V', 'E030M', 'E030V',
         'E031M', 'E031V', 'E033M', 'E033V', 'E034M', 'E034V', 'E036M', 'E036V', 'E037M', 'E037V', 'E038M']
files2 = ['E038V', 'E039M', 'E039V', 'E042M', 'E042V', 'E045M', 'E045V', 'E046M', 'E046V', 'E048M', 'E048V',
          'E052M', 'E052V', 'E054M', 'E054V', 'E060M', 'E060V', 'E062M', 'E062V', 'E065M', 'E065V', 'E066M', 'E067M']
# fileName = sys.argv[1] if len(sys.argv) > 1 else ""
for fileName in files2:
    filePath = "../../videos/%s.mp4" % fileName
    print(fileName)
    if (os.path.exists(filePath)):
        try:
            os.makedirs("./frames/%s" % fileName)
        except OSError as e:
            shutil.rmtree("./frames/%s" % fileName)
            os.makedirs("./frames/%s" % fileName)
            print("directory exists")
        videoStreamer(fileName)
    else:
        print("File name %s doesnt exist" % fileName)
