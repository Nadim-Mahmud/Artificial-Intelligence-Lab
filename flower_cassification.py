import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

DIR = "D:\\Experiment\\dataset\\"

def main():
    prepare_image_array("zinia", 256)
    prepare_image_array("scarlet", 256)


def prepare_image_array(PATH,  IMG_HEIGHT, IMG_WIDTH):

    imDir = DIR + PATH + "\\"
    imList = os.listdir(imDir)
    #print(imList)


    imSet = []
    n = len(imList)
    for i in range(n):

        imPath = imDir + imList[i]
        if(os.path.exists(imPath)):

            # load image
            print(imPath)
            img = cv2.imread(imPath)
            print(img.shape)

            # resize image
            # onpen cv different size format
            resizeImg = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) # bi qubic interpolation
            # print(resizeImg.shape)

            # convert BGR image into RGB image
            rgbImg = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2RGB)

            #cv2.imwrite(DIR + "new\\" + PATH + str(i)+ ".jpg" , rgbImg)

            # put image into a list
            imSet.append(rgbImg)

        else:
            print("It is not a valid path")

    print(len(imSet))

    imSet = np.array(imSet, dtype = np.uint8)
    print(imSet.shape)

    #display_data(imSet[:9])

    return imSet



def display_data(imSet):
    plt.figure(figsize = (20,20))
    
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(imSet[i], cmap='gray')

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()