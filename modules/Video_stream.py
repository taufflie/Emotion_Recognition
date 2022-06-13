import cv2
from face_detection import face_detection



def saveVideoFrames():

    cap = cv2.VideoCapture(capture)
    i = 1
    while (cap.isOpened() ):
        ret, frame = cap.read()
        if ret == False:
            break
        face = face_detection(frame)
        face = cv2.resize(face, (48, 48))
    cap.release()
    cv2.destroyAllWindows()
    return

def main():
    video_path = '001.mp4'
    saveVideoFrames(video_path)
    return

if __name__ == '__main__':
    main()