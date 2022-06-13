import cv2
from matplotlib import pyplot as plt

def face_detection(img,detector):
    context = 0.3

    results = detector.detect_faces(img)
    if results:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height

        x1 = x1 - int(context * width)
        x2 = x2 + int(context * width)
        y1 = y1 - int(context * height)
        y2 = y2 + int(context * height)

        face = img[y1:y2, x1:x2]

        return face
    else:
        return None
