import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from face_detection import face_detection
from keras import backend as K
#####################################################################################################################
#####################################################################################################################
detector = MTCNN()

# classes = ["anger", "happiness", "neutral", "sadness", "surprise"]
classes = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
# classes = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
# classes = ["NF", "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise", "unknown"]

#####################################################################################################################
#####################################################################################################################

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

#####################################################################################################################
#####################################################################################################################
def imagePreprocessing(base_directory):

    test_directory = base_directory + '/FER2013' + 'Test'

    #TODO - Create the image data generators for train and validation
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(test_directory, target_size=(48, 48), batch_size=50, class_mode='categorical', color_mode='grayscale', shuffle=False)

    #TODO - Analize the output of the train and validation generators
    for data_batch, labels_batch in test_generator:
        print('Data batch shape in train: ', data_batch.shape)
        print('Labels batch shape in train: ', labels_batch.shape)
        break

    return test_generator

#####################################################################################################################
#####################################################################################################################
def Predict_Stream(model):

    cap = cv2.VideoCapture(0)
    counter = 1

    # TODO - Initialize the vector storing the frames
    X_train = np.zeros((48, 48))
    X_train = X_train.reshape(1, 48, 48, 1)

    while (True):

        # TODO - Capture frame-by-frame
        ret, frame = cap.read()

        # TODO - Perform face detection and change size, color of image
        image = face_detection(frame, detector)
        face = cv2.resize(image, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.reshape(1, 48, 48, 1)

        if counter ==1:
            X_train = face

        if counter !=1:
            X_train = np.append(X_train, face, axis=0)

        # TODO - Load model and make the predictions
        if counter >= 10:

            scores = model.predict(X_train)
            y_pred = np.argmax(scores, axis=1)
            y_pred = np.bincount(y_pred).argmax()
            # y_pred = y_pred[-1]

            # TODO - Add the emotion to the frame and display it
            cv2.putText(frame, classes[y_pred], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow('frame', frame)

            X_train = np.delete(X_train, 0, 0)
        counter +=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # TODO - When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return

#####################################################################################################################
#####################################################################################################################

def Create_Confusion_Matrix(tensor, model):

    # TODO - Predict the emotion for each frame
    # accuracy = model.evaluate(tensor)
    scores = model.predict(tensor)
    predicted_class_indices = np.argmax(scores, axis=1)

    labels = (tensor.classes)

    # TODO - Normalize the Matrix

    cm = confusion_matrix(labels, predicted_class_indices)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # TODO - Display the Matrix
    print(classification_report(labels, predicted_class_indices, target_names=classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Reds)
    plt.title("Confusion Matrix")
    plt.show()

    return

#####################################################################################################################
#####################################################################################################################
def Predict_Video(model):

    # TODO - Go through each video file
    for filename in glob.glob('./videos/*'):

        # TODO - Open one video
        cap = cv2.VideoCapture(filename)

        # TODO - Create a new directory for this video
        filename = filename[:-4]
        filename = filename[8:]
        os.makedirs('./VideoFrames_disgust/{}'.format(filename), exist_ok=True)

        # TODO - Initialize the vector storing the frames
        counter = 1
        X_train = np.zeros((48, 48))
        X_train = X_train.reshape(1, 48, 48, 1)
        while (cap.isOpened()):

            # TODO - Capture frame-by-frame and exit if video is over
            ret, frame = cap.read()
            if ret == False:
                break

            # TODO - Perform face detection and change size, color of image
            image = face_detection(frame, detector)
            face = cv2.resize(image, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face.reshape(1, 48, 48, 1)

            if counter ==1:
                X_train = face

            # TODO - Add the frames to the array until there are 10 frames in it
            if counter !=1:
                X_train = np.append(X_train, face, axis=0)

            if counter >= 10:

                # TODO - Make the predictions

                scores = model.predict(X_train)
                y_pred = np.argmax(scores, axis=1)
                y_pred = np.bincount(y_pred).argmax()
                # y_pred = y_pred[-1]

                # TODO - Add the emotion to the frame and display it
                cv2.putText(frame, classes[y_pred], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                ordered_counter = str(counter).zfill(4)
                path = './VideoFrames_disgust/' + filename + '/' + ordered_counter + '.jpg'
                cv2.imwrite(path, frame)
                print('Saved frame n°{}'.format(ordered_counter))

                X_train = np.delete(X_train, 0, 0)
            counter +=1
        # TODO - When everything done, release the capture
        cap.release()

    return

#####################################################################################################################
#####################################################################################################################

def read_predicted_video(directory_name):

    # TODO - initialize the list containing all frames
    img_array = []

    # TODO - Go through each video file
    for filename in glob.glob('./VideoFrames_disgust/' + directory_name + '/*'):

        # TODO - Read all of the images for one file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # TODO - Store all of these images into the array
        img_array.append(img)

    # TODO - Create a Video Writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./predicted_videos/' + directory_name + '.avi', fourcc, 17, size)

    # TODO - Save the images to a video file
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return

#####################################################################################################################
#####################################################################################################################

def main():


    # TODO - Load the model

    base_directory = "FERPlus-master/7_classes_enhanced"
    model = tf.keras.models.load_model('./models/7_classes_enhanced_Ep_60.h5', custom_objects = {'focal_loss_fixed': focal_loss()})

    # TODO - Create tensor using the test set
    test_generator = imagePreprocessing(base_directory)

    # TODO - Call the function that do the prediction for the video stream
    # Predict_Stream(model)

    # TODO - Call the function that do the prediction for the pre-recorded video
    # Predict_Video(model)

    # TODO - Read the video with prediction
    # read_predicted_video('Daniel_stable')

    # TODO - Call the function that create the confusion matrix
    Create_Confusion_Matrix(test_generator, model)

    # # TODO - Create saliency map
    # map = get_saliency_map(model, './0077', 4)
    # cv2.imshow(map)
    return

#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':
    main()