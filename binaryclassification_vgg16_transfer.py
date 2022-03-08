from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

def main():
    newModel = build_model()
    imgSet = prepare_image_data()
    labelSet = perpare_label_data()

    trainImage = imgSet[:70]
    trainLabel = labelSet[:70]
    testImage = imgSet[70:]
    testLabel = labelSet[70:]

    newModel.compile(loss = 'mse', optimizer = 'rmsprop')

    newModel.fit(trainImage, trainLabel, batch_size = 8, epochs = 100, validation_split=0.05, callback = EarlyStopping(monitor = 'val_loss', patience = 10))
    
    newModel.compile(metric = 'accuracy')
    predict = newModel.predict(testImg[:10])
    print(predict)
    newModel.evaluate()

def perpare_label_data():
    # perpare labels for two classes
    # 1: Chair
    # 0: table

    # Convert labels into One-hot vectors

    # return one-hot vectors

def prepare_image_data():
    # Load image

    # Resize images

    # put images into a 4D numpy array
    # (n, 256, 256, 3)

    # preprocess image according to VGG16

    #return 4D image set

def build_model():
    baseModel = VGG16(input_shape = (256, 256, 3), include_top = False)
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(16, 'sigmoid')(x)
    outputs = Dense(2, 'sigmoid')(x)
    model = Model(inputs, outputs)
    model.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    model.summary()

    return model

if __name__ == '__main__':
    main()