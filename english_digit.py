
import numpy as np

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, History


DIR = 'D:\\Experiment\\model.hdf5'

def main():
    modelPath = DIR
    #model = train_model()
    test_model()


def test_model():

    trainX, trainY, testX, testY = perpare_data()

    model = load_model(DIR)

    print('\nAccuracy section: ')
    model.compile(metrics = 'accuracy')
    model.evaluate(testX, testY)
    print('\n')

    predictY =  model.predict(testX[:5])
    print(predictY)
    print(testY[:5])
    for i in range(5):
        print(np.argmax(predictY[i]))


def train_model():

    trainX, trainY, testX, testY = perpare_data()
   
    model = build_model()

    callbackList = [EarlyStopping(monitor = 'val_loss', patience = 10), History()]
    model.fit(trainX, trainY, epochs = 500, callbacks = callbackList, validation_split = 0.2)

    model.compile(metrics = 'accuracy')
    model.evaluate(testX, testY)


    predictY =  model.predict(testX[:5])
    print(predictY)
    print(testY[:5])
    for i in range(5):
        print(np.argmax(predictY[i]))


    #save the model
    model.save(DIR)

    return model

def build_model():
    
    inputs = Input((28,28))
    x = Flatten()(inputs)
    x = Dense(16, activation = 'sigmoid')(x)
    x = Dense(8, activation = 'sigmoid')(x)
    outputs = Dense(10, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.summary()

    model.compile(loss = 'mse', optimizer = 'rmsprop')

    return model


def perpare_data():
    (trainX, trainY), (testX, testY) = load_data()
    print(trainX.shape, trainY.shape, testX.shape, testY)

    # converting reange inside one
    print(trainX.dtype, trainX.max(), trainX.min())
    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    trainX /= 255
    testX /= 255
    print(trainX.dtype, trainX.max(), trainX.min())

    print(trainY[:5])

    #converting numeric values 0,1,2....9 into one hot vectores
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    print(trainY[:5])

    return  trainX, trainY, testX, testY

if __name__ == '__main__':
    main()