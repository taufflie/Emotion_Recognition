import cv2
from face_detection import face_detection



def saveVideoFrames(capture):
    pathFolderWhereToSaveFrames = ".//VideoFrames_disgust//"
    cap = cv2.VideoCapture(capture)
    #frame_per_second = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    i = 1
    while (cap.isOpened() ):
        ret, frame = cap.read()
        if ret == False:
            break
        ordered_i = str(i).zfill(4)
        face = face_detection(frame)
        face = cv2.resize(face, (48,48))
        cv2.imwrite(pathFolderWhereToSaveFrames + ordered_i + '.jpg', face)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return

def main():
    video_path = '001.mp4'
    saveVideoFrames(video_path)
    return

if __name__ == '__main__':
    main()