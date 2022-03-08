from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

m = 28
n = 24
c = 10

def main():

   model = build_model()
   model.compile(loss='mse', metrics='accuracy')


def build_model():
   
   inputs = Input((m,n))
   x = Flatten()(inputs)
   x = Dense(4, activation='sigmoid')(x)
   x = Dense( 8, activation='sigmoid')(x)
   x = Dense(4, activation='sigmoid')(x)
   outputs = Dense(c)(x)
  
   model = Model(inputs, outputs)
   model.summary()

   return model

if __name__ == '__main__':
   main()
