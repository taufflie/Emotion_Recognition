import cv2
import os
from face_detection import face_detection
from mtcnn import MTCNN

#####################################################################################################################
#####################################################################################################################

def saveVideoFrames(capture,detector):

    cap = cv2.VideoCapture(capture)
    i = 1
    while cap.isOpened():
        # TODO - Capture frame-by-frame and exit if video is over
        ret, frame = cap.read()
        if ret == False:
            break
        ordered_i = str(i).zfill(4)
        face = face_detection(frame, detector)
        if (face is not None):
            face_resized = cv2.resize(face, (48, 48))
            face_grey = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(r"../Frames/" + capture + ordered_i + '.jpg', face_grey)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return

#####################################################################################################################
#####################################################################################################################

def create_images(detector):

    for file in os.listdir():
        saveVideoFrames(file, detector)
    return

#####################################################################################################################
#####################################################################################################################


def main():
    detector = MTCNN()
    folder = r"CREMAD/Videos"
    os.chdir(folder)
    create_images(detector)
    return

if __name__ == '__main__':
    main()