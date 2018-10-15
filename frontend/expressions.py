import cv2
import numpy as np
from os import listdir, path
from functools import reduce
import dlib
import glob

detector = dlib.get_frontal_face_detector()
# TODO: fix path
predictor = dlib.shape_predictor(
    "../frames-to-landmarks/shape_predictor_68_face_landmarks.dat")

files = listdir()
lastIndex = 0
files = glob.glob('./*.avi')
if (len(files) > 0):
    lastIndex = int(max(files, key=path.getctime).split("-")[1].split(".")[0])
lastIndex += 1


def camera(index):
    print(index)
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('interview-%s.avi' % str(index), cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    while(True):
        ret, frame = cap.read()

        if ret == True:

            # Write the frame into the file 'output.avi'
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def analyzeFrames(fileName):
    print(fileName)
    data = open("./test.csv", "a")
    stream = cv2.VideoCapture(str(fileName))
    FPS = stream.get(cv2.CAP_PROP_FPS)
    index = 0
    while True:
        (grabbed, frame) = stream.read()
        if not grabbed:
            return
        if((index % 3 == 0)):
            # TODO analyze frame
            rects = detector(frame, 0)
            face = dlib.get_face_chip(frame,  predictor(frame, rects[0]), 48)
            gray_frame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            new_image = ""
            for x in range(48):
                for y in range(48):
                    new_image += str(gray_frame[x][y]) + " "
            data.write(new_image + "\n")
        index += 1
    stream.release()
    data.close()


camera(lastIndex)
analyzeFrames("./interview-%s.avi" % str(lastIndex))
