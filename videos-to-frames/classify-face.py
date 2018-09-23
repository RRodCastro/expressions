from os import listdir
import cv2
from time import sleep
H = [
    'E002M', 'E002V', 'E003M', 'E003V', 'E006M', 'E006V',
    'E010M', 'E010V', 'E013M', 'E013V', "E015V", "E015M",
    "E036M", "E036V", "E037V", "E037M", 'E039M', 'E039V',
    "E042V", "E042M", "E048M", "E048V", 'E052M', 'E052V',
    'E063M', 'E063V', 'E064M',  "E066M", 'E066V', 'E068M',
    'E068V']

MDONE = []

M = [
    'E014M', 'E014V', 'E016M', 'E016V', 'E018M', 'E018V',
    'E026M', 'E026V', 'E027M', 'E027V', 'E030M', 'E030V',
    'E031M', 'E031V', 'E033M', 'E033V', 'E034M', 'E034V',
    'E038M', 'E038V', 'E045M', 'E045V', 'E046M', 'E046V',
    'E054M', 'E054V', 'E060M', 'E060V', 'E062M', 'E062V',
    'E065M', 'E065V', 'E067M', 'E067V']

data = open("classified-frames.csv", "a")


for interview in M:
    if("V" in interview):
        continue
    print("current interview ... " + interview)
    frames = listdir("../videos-to-frames/beforeAnswer/%s" % interview)
    frames = sorted(frames, key=lambda name: int(name.split("-")[0]))
    for frame in frames:
        gray = cv2.imread(
            "./beforeAnswer/%s/%s" % (interview, frame))
        cv2.imshow(interview, gray)
        test = cv2.waitKey(0)
        if (test == ord('0')):
            face = 0
        elif (test == ord('1')):
            face = 1
        else:
            face = 2

        data.write(interview + ',' + frame + ',' + str(face) + '\n')
    cv2.destroyAllWindows()
