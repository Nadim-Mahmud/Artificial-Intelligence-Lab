from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from flower_cassification import prepare_image_array
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.utils import to_catagorical

DIR = "D:\\Experiment\\dataset\\"
IMG_HEIGHT = 32
IMG_WIDTH = 32

def main():
    #build_model()
    preprocess_data()


def preprocess_data():

    #load image data
    imSet1 = prepare_image_array("zinia", IMG_HEIGHT, IMG_WIDTH)
    n = imSet1.shape[0]

    imSet2 = prepare_image_array("scarlet", IMG_HEIGHT, IMG_WIDTH)
    m = imSet1.shape[0]

    imSet = np.concatenate((imSet1, imSet2), axis = 0)
    print(imSet.shape)

    # convering to vgg16
    imSet = preprocess_input(imSet)


    #repare labels
    labelSet1 = np.zeros(n, dtype=np.uint8)
    labelSet2 = np.zeros(m, dtype=np.uint8)
    labelSet = np.concatenate((labelSet1, labelSet2))
    print(labelSet)

    # converting to one hot vector
    labelSet = to_catagorical(labelSet)
    print(labelSet)

    # Shuffle image data and labels
    n = imSet.shape[0]
    indices = np.arange(n)

    random.

    


def build_model():
    baseModel = VGG16(input_shape(IMG_HEIGHT, IMG_WIDTH, 3), include_top = False)
    baseModel.summary()

    for layer in baseModel.layers:
        layer.trainable = False
    
    baseModel.summary()

    inputs = baseModel.input
    x = baseModel.output
    x = Flatten()(x)
    x = Dense(16, activation = 'sigmoid')(x)
    outputs = Dense(2, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.summery()

    return model


if __name__ == "__main__":
    main()