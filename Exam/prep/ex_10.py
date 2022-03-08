from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

m = 512
n = 512
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

model.compile(loss='mse', metrics='accuracy')


