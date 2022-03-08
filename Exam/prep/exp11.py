from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical

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

trainY = to_categorical(trainY==1)
testY = to_categorical(testY==1)

print(trainY[:9])

trainX = trainX.astype(np.float32)
testX = testX.astype(np.float32)

trainX = trainX/255
testX = testX/255

print(trainX.max())


#Creating model

m = 28
n = 28
h = 4
c = 2

inputs = Input((m,n,1))

x = Conv2D(filters=4, kernel_size=(2,2), padding='same')(inputs)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(filters=4, kernel_size=(2,2), strides=(2,2), padding='same')(x)
x = Conv2D(filters=4, kernel_size=(2,2), strides=(2,2), padding='same')(x)


x = Flatten()(x)
outputs = Dense(2)(x)

model = Model(inputs, outputs)

model.summary()

model.compile(loss='mse', optimizer='rmsprop')


model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.2)

predictY = model.predict(testX)


for i in range(9):
    y = np.argmax(testY[i])
    py = np.argmax(predictY[i])
    print('original : {} predicted : {}'.format(y, py))

model.compile(metrics=['accuracy'])
model.evaluate(testX, testY)
