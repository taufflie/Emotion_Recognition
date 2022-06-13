import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import tensorflow as tf
from keras import backend as K
from keras_video import VideoFrameGenerator
import glob
from matplotlib import pyplot
from tensorflow.keras.layers import Flatten, Dense, TimeDistributed, Input, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import time
#####################################################################################################################
#####################################################################################################################

# classes = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
dict_classes = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
# classes = ["anger", "fear", "happiness", "neutral", "sadness"]
num_classes = len(dict_classes)

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
def LSTM(num_classes):
    basemodel = tf.keras.models.load_model('./Models/Fixed_5_classes_Lr_0.01_L2_Softmax_Ep_200.h5')
    # basemodel = tf.keras.models.load_model('./Models/7_classes_enhanced_Ep_60.h5')
    basemodel.summary()

    modelPart = basemodel.layers[-2].output
    # modelPart = Flatten()(modelPart)
    cnn = tf.keras.Model(inputs=basemodel.input, outputs=modelPart)

    inputs = tf.keras.Input(shape=(10, 48, 48, 1))
    encoded_frames = TimeDistributed(cnn)(inputs)

    encoded_sequence = GRU(units=128, return_sequences=True, dropout=0.3)(encoded_frames)
    encoded_sequence = GRU(units=128, return_sequences=False, dropout=0.3)(encoded_sequence)

    outputs = Dense(num_classes, activation="softmax",
                                kernel_initializer='orthogonal',
                                use_bias=True, trainable=True,
                                kernel_regularizer=l2(0.001),
                                bias_regularizer=l2(0.001),
                                name='myPrediction')(encoded_sequence)

    model = tf.keras.Model([inputs], outputs)

    loss = focal_loss()
    opt = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])

    model.summary()

    return model

#####################################################################################################################
#####################################################################################################################
def data_generator():

    # TODO - use sub directories names as classes
    classes = [i.split(os.path.sep)[1] for i in glob.glob('CREMAD/Videos_classes/*')]
    classes.sort()

    # some global params
    SIZE = (48, 48)
    CHANNELS = 1
    NBFRAME = 10
    BS = 50

    # TODO - pattern to get videos and classes
    glob_pattern = 'CREMAD/Videos_classes/{classname}/*'

    # TODO- Create video frame generator
    train_generator = VideoFrameGenerator(
        classes=classes,
        glob_pattern=glob_pattern,
        nb_frames=NBFRAME,
        split=.2,
        shuffle=True,
        batch_size=BS,
        target_shape=SIZE,
        nb_channel=CHANNELS,
        use_frame_cache=False)

    validation_generator = train_generator.get_validation_generator()

    #TODO - Analyze the output of the train and validation generators
    for data_batch, labels_batch in train_generator:
        print('Data batch shape in train: ', data_batch.shape)
        print('Labels batch shape in train: ', labels_batch.shape)
        break
    for data_batch, labels_batch in validation_generator:
        print('Data batch shape in validation: ', data_batch.shape)
        print('Labels batch shape in validation: ', labels_batch.shape)
        break

    return train_generator, validation_generator

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
def main():
    start_time = time.time()

    # base_directory = "FERPlus-master/7_classes_enhanced"
    # name_dataset = base_directory[15:]
    #TODO - Set the number of epochs
    Number_epochs = 60

    #TODO - Set the name of the model and the plots
    # filename = "{}".format(name_dataset)+'_Ep_'+"{}".format(Number_epochs)
    filename = 'LSTM_Ep_'+"{}".format(Number_epochs)

    #TODO - Call the imagePreprocessing method
    train_generator, validation_generator = data_generator()

    #TODO - Call the method that creates the CNN model
    model = LSTM(num_classes)

    #TODO - Train the model

    history = model.fit(train_generator, steps_per_epoch=99, epochs=Number_epochs, validation_data=validation_generator, validation_steps=68)
    model.save(filename+'.h5')

    # TODO - Visualize the performances
    visualizeTheTrainingPerformances(history, filename)
    print("--- %s seconds ---" % (time.time() - start_time))

    return
#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':
    main()

#####################################################################################################################
#####################################################################################################################
