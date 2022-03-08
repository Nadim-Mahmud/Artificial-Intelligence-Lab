from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

m = 3
n = 3
h = 4
c = 2

inputs = Input((m,n))
x = Flatten()(inputs)
x = Dense(8, activation = 'sigmoid')(x)
x = Dense(16, activation = 'sigmoid')(x)
x = Dense(8, activation = 'sigmoid')(x)
outputs = Dense(2)(x)

model = Model(inputs, outputs)

model.summary()

model.compile(loss='mse', metrics='accuracy')


