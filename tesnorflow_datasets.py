from tensorflow.keras.datasets import mnist, fashion_mnist, boston_housing, cifar10
import matplotlib.pyplot as plt

def main():

    (trainX, trainY), (testX, testY) = mnist.load_data()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    #display_data(trainX[:9])

    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    #display_data(trainX[:9])

    
    (trainX, trainY), (testX, testY) = boston_housing.load_data()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    #display_data(trainX[:9])


    (trainX, trainY), (testX, testY) = cifar10.load_data()
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    #display_data(trainX[:9])



def display_data(imSet):
    plt.figure(figsize = (20,20))
    
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(imSet[i], cmap='gray')

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()