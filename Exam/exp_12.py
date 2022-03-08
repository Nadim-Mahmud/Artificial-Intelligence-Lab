from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.utils import to_categorical
import random
import os
import cv2
import matplotlib.pyplot as plt


def main():
	# original_model_prediction()

	trainX, trainY, testX, testY = preprocess_data()

	model = build_model()

	# Train model.

	callbackList = [EarlyStopping(monitor='val_loss', patience=10), History()]
	history = model.fit(trainX, trainY, epochs=5, batch_size=16,
						callbacks=callbackList, validation_split=0.2)

	# Check what the model predicts.
	predictY = model.predict(testX)
	for i in range(10):
		y = np.argmax(testY[i])
		pY = np.argmax(predictY[i])
		print('Original Y: {}, Predicted Y: {}'.format(y, pY))

	# Estimate the performance of the NN.
	model.compile(metrics='accuracy')
	model.evaluate(testX, testY)



def resize(data):
	idx = 0
	tmpX = []
	for image in data:
		tmpX.append(cv2.resize(image, (32, 32)))
	data = np.array(tmpX, dtype = np.uint8)

	return data


def preprocess_data():
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

	print(trainX.shape, trainY.shape, testX.shape, testY.shape)

	trainX = resize(trainX)
	testX = resize(testX)

	print(trainX.shape, trainY.shape, testX.shape, testY.shape)

	print(trainY[:9])
	#display_images_with_predictions(trainX[:9], trainY[:9])

	index = np.argwhere((trainY == 0) | (trainY == 1))
	index = index[:, 0]

	trainX = trainX[index]
	trainY = trainY[index]

	index = np.argwhere((testY == 0) | (testY == 1))
	index = index[:, 0]

	testX = testX[index]
	testY = testY[index]

	print(trainX.shape, trainY.shape)
	print(testX.shape, testY.shape)

	#plot_digits(trainX[:9], trainY[:9])

	#print('DataY : {}',format(trainY[:100]))

	# Convert numeric digit labels into one-hot vectors.
	# 0: 1 0 0 0 0 0 0 0 0
	# 1: 0 1 0 0 0 0 0 0 0
	# 2: 0 0 1 0 0 0 0 0 0
	print('Labels: {}, DataType: {}'.format(
		trainY[:10], trainY[:10].dtype))

	trainY = to_categorical(trainY, )
	testY = to_categorical(testY, )

	#trainY = trainY[0, :, :]
	#testY = testY[0, :, :]

	print('Labels: {}, DataType: {}'.format(
		trainY[:10], trainY[:10].dtype))

	# To convert pixel values from 0-255 into 0-1.
	print('DataType: {}, Max: {}, Min: {}'.format(
		trainX.dtype, trainX.max(), trainX.min()))
	trainX = trainX.astype(np.float32)
	testX = testX.astype(np.float32)
	trainX /= 255
	testX /= 255
	print('DataType: {}, Max: {}, Min: {}'.format(
		trainX.dtype, trainX.max(), trainX.min()))


	trainX = np.stack((trainX,)*3, axis=-1)
	testX = np.stack((testX,)*3, axis=-1)
	print(trainX.shape, testX.shape)

	return trainX, trainY, testX, testY


def build_model():
    baseModel = VGG16(input_shape=(32, 32, 3), include_top=False)
    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(8, activation='sigmoid')(x)
    outputs = Dense(2, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()

    model.compile(loss='mse', optimizer='rmsprop')

    return model


def original_model_prediction():
    # Load a pre-trained model.
    model = VGG16()
    model.summary()

    # Prepare data set.
    _, _, testX, testY = preprocess_data(224, 224)

    # Predict which class the loaded image belongs to
    # List of 1000 classes: http://image-net.org/challenges/LSVRC/2014/browse-synsets
    predictions = model.predict(testX)
    predictions = decode_predictions(predictions, top=1)

    n = testX.shape[0]
    predictedClass = []
    for i in range(n):
        className = predictions[i][0][1]
        predictedClass.append(className)

    # Draw some example images with predictions
    display_images_with_predictions(testX, predictedClass)


def display_images_with_predictions(imgSet, labelSet):
    plt.figure(figsize=(20, 20))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title(labelSet[i])
        plt.imshow(imgSet[i])
        plt.axis('off')
    plt.show()



if __name__ == '__main__':
    main()
