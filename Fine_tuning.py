import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2
import time
#####################################################################################################################
#####################################################################################################################

classes = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
num_classes = len(classes)

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
def CNN1_model_fine_tuned(num_classes):

    basemodel = tf.keras.models.load_model('./Models/Fixed_5_classes_Lr_0.01_L2_Softmax_Ep_200.h5')
    basemodel.summary()

    # for layer in basemodel.layers[:-2]:
    #     layer.trainable = False

    model = tf.keras.models.Sequential(basemodel.layers[:-1])
    model.summary()

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001), name='softmax'))

    # loss = tfa.losses.SigmoidFocalCrossEntropy()
    loss = focal_loss()

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

#####################################################################################################################
#####################################################################################################################
def visualizeTheTrainingPerformances(history, fname):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    pyplot.title('Training and validation accuracy')
    pyplot.plot(epochs, acc, 'bo', label='Training accuracy')
    pyplot.plot(epochs, val_acc, 'b', label='Validation accuracy')
    pyplot.legend()
    pyplot.xlabel("Number of epochs")
    pyplot.ylabel("Accuracy")

    pyplot.savefig('./Curves/Accuracy_{}.png'.format(fname))

    pyplot.figure()
    pyplot.title('Training and validation loss')
    pyplot.plot(epochs, loss, 'bo', label='Training loss')
    pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
    pyplot.legend()
    pyplot.xlabel("Number of epochs")
    pyplot.ylabel("Loss")
    pyplot.savefig('./Curves/Loss_{}.png'.format(fname))

    pyplot.show()

    return

#####################################################################################################################
#####################################################################################################################
def imagePreprocessing(base_directory):

    train_directory = base_directory +'/FER2013' + 'Train'
    validation_directory = base_directory + '/FER2013' + 'Valid'
    test_directory = base_directory + '/FER2013' + 'Test'

    #TODO - Create the image data generators for train and validation
    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_directory, target_size=(48, 48), batch_size=128, class_mode='categorical', color_mode='grayscale')
    validation_generator = validation_datagen.flow_from_directory(validation_directory, target_size=(48, 48), batch_size=50, class_mode='categorical', color_mode='grayscale')
    test_generator = test_datagen.flow_from_directory(test_directory, target_size=(48, 48), batch_size=50, class_mode='categorical', color_mode='grayscale')

    #TODO - Analize the output of the train and validation generators
    for data_batch, labels_batch in train_generator:
        print('Data batch shape in train: ', data_batch.shape)
        print('Labels batch shape in train: ', labels_batch.shape)
        break
    for data_batch, labels_batch in validation_generator:
        print('Data batch shape in validation: ', data_batch.shape)
        print('Labels batch shape in validation: ', labels_batch.shape)
        break
    for data_batch, labels_batch in test_generator:
        print('Data batch shape in train: ', data_batch.shape)
        print('Labels batch shape in train: ', labels_batch.shape)
        break

    return train_generator, validation_generator, test_generator

#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
def main():
    start_time = time.time()

    base_directory = "FERPlus-master/7_classes_enhanced"
    name_dataset = base_directory[15:]
    #TODO - Set the number of epochs
    Number_epochs = 60

    #TODO - Set the name of the model and the plots
    filename = "{}".format(name_dataset)+'_Ep_'+"{}".format(Number_epochs)

    #TODO - Call the imagePreprocessing method
    train_generator, validation_generator, test_generator = imagePreprocessing(base_directory)

    #TODO - Call the method that creates the CNN model
    model = CNN1_model_fine_tuned(num_classes)

    #TODO - Train the model

    history = model.fit(train_generator, steps_per_epoch=106, epochs=Number_epochs, validation_data=validation_generator, validation_steps=68)
    accuracy = model.evaluate(test_generator)
    print("The testing loss is: {} and the testing accuracy is: {} %".format(round(accuracy[0], 3), 100 * round(accuracy[1], 3)))
    model.save(filename+'.h5')
    # TODO - Visualize the performances
    visualizeTheTrainingPerformances(history, filename)
    print("--- %s seconds ---" % (time.time() - start_time))
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
