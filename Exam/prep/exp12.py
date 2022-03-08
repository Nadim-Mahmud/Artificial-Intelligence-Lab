from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt


DIR = r'D:\Experiment\Exam\prep'


(trainX, trainY), (testX, testY) = mnist.load_data()
print(trainX.shape, trainY.shape)

indices = np.argwhere((trainY==1)|(trainY==2))
indices = np.squeeze(indices)

trainX = trainX[indices]
trainY = trainY[indices]

print(trainX.shape)

indices = np.argwhere((testY==1)|(testY==2))
indices = np.squeeze(indices)

testX = testX[indices]
testY = testY[indices]

print(trainY[:9])

plt.imshow(trainX[0])
plt.show()


trainY = to_categorical(trainY==1)
testY = to_categorical(testY==1)

print(trainY[:9])

trainX = trainX.astype(np.float32)
testX = testX.astype(np.float32)

trainX = trainX/255
testX = testX/255

print(trainX.max())

# reshaping the image size
tmp = []

for image in trainX:
    image = cv2.resize(image, (32,32))
    tmp.append(image)

trainX = np.array(tmp)

tmp = []
for image in testX:
    image = cv2.resize(image, (32,32))
    tmp.append(image)

testX = np.array(tmp)

trainX = np.stack((trainX,)*3, axis=-1)
testX = np.stack((testX,)*3, axis=-1)

print(testX.shape, testX.dtype)


#Creating model

m = 28
n = 28
h = 4
c = 2

basemodel = VGG16(input_shape=(32,32,3), include_top=False)
basemodel.summary() 

for layer in basemodel.layers:
    layer.trainable = False

basemodel.summary()


inputs = basemodel.input
x = basemodel.output
x = Flatten()(x)
x = Dense(8, activation='sigmoid')(x)
outputs = Dense(2, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.summary()

model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.0001))

callbackList = [EarlyStopping(monitor='val_loss', patience=10), History()]
model.fit(trainX, trainY, epochs=1, batch_size=16, callbacks=callbackList, validation_split=0.2)

model.save(DIR + '\exp12.hdf5')


predictY = model.predict(testX)


for i in range(9):
    y = np.argmax(testY[i])
    py = np.argmax(predictY[i])
    print('original : {} predicted : {}'.format(y, py))

model.compile(metrics=['accuracy'])
model.evaluate(testX, testY)


md = load_model(DIR+'\\'+'exp12.hdf5')
md.summary()

