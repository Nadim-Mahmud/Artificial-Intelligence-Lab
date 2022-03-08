from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
 
m = 6
n = 6
c = 10

def main():
   model = build_model()
   

 
def build_model():
   # create layers
   inputs = Input((m,n,1))
   # #assume hidden layer h=3
   x = Conv2D(filters = 3, kernel_size = (2,2), strides = (2,2),padding = 'valid',use_bias = False)(inputs)
   x = Conv2D(filters = 3, kernel_size = (2,2), strides = (2,2),padding = 'valid',use_bias = False)(x)
   x = Flatten()(x)
   outputs = Dense(c)(x)
   # create model
   model = Model(inputs, outputs)
   model.summary()
 
   return model

if __name__ == '__main__':
   main()
