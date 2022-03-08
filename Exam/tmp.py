from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2


def resize(data):
	tmp = []
	for img in data:
		img = cv2.resize(img, (32,32))
		tmp.append(img)
	data = np.array(tmp)
	return data


(trainX, trainY), (testX, testY) =  mnist.load_data()

train_indices = np.argwhere((trainY == 2) | (trainY == 3))
test_indices = np.argwhere((testY == 2) | (testY == 3))

train_indices = np.squeeze(train_indices)
test_indices = np.squeeze(test_indices)

trainX = trainX[train_indices]
trainY = trainY[train_indices]

testX = testX[test_indices]
testY = testY[test_indices]

#trainY = to_categorical(trainY == 2)
#testY = to_categorical(testY == 2)
trainY = (trainY==2)
testY = (testY == 2)

print(trainY[:9])

trainX = trainX.astype(np.float32)
testX = testX.astype(np.float32)

trainX = trainX/255
testX = testX/255

trainX = np.pad(trainX, ((0,0),(2,2),(2,2)), 'constant')
testX = np.pad(testX, ((0,0),(2,2),(2,2)), 'constant')

trainX = np.stack((trainX,)*3, axis = -1)
testX = np.stack((testX,)*3, axis = -1)

print(trainX.shape, trainY.shape, testX.shape, testY.shape)


baselayer = VGG16(input_shape=(32,32,3), include_top=False)

for layer in baselayer.layers:
	layer.trainable = False
baselayer.summary()


inputs = baselayer.input
x = baselayer.output
x = Flatten()(x)
x = Dense(8, activation='sigmoid')(x)

outputs  = Dense(1)(x)


model = Model(inputs, outputs)

model.summary()

model.compile(loss='mse', optimizer='rmsprop')
model.fit(trainX, trainY, epochs = 5, batch_size = 16, validation_split = 0.2)


predictY = model.predict(testX)

print(predictY[:9])


for i in range(10):
	y = np.argmax(testY[i])
	py = np.argmax(predictY[i])
	print('origianl : {} predicted : {}'.format(testY[i],py))
	if(predictY[i] > .5):
		print('2')
	else:
		print('3')


model.compile(metrics = ['accuracy'])
model.evaluate(testX, testY)


