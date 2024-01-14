import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.python.keras.optimizers import adam_v2



def forName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names = columns)
    data['Center'] = data['Center'].apply(forName)
    print(data.shape[0])
    return data

def balanceData(data,display=True):
    nBins = 31
    samplesPerBin = 1000
    hist,bins = np.histogram(data['Steering'],nBins)
    #print(bins)
    center = (bins[:-1] + bins[1:])*0.5
    #print(center)
    plt.bar(center,hist,width = 0.06)
    plt.plot((-1,1),(samplesPerBin,samplesPerBin))
    #   plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] >= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print(len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print(len(data))

    #if display:
        #hist, _  = np.histogram(data['Steering'], nBins)
        #plt.bar(center, hist, width=0.06)
        #plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        #plt.show()

    return data

def loadData(path,data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG',indexedData[0]))
        steering.append(float(indexedData[-1]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath,steering

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    #panning the image
    if np.random.rand()<0.5:
        pan = iaa.Affine(translate_percent = {'x': (-0.1,0.1),'y': (-0.1,0.1)})
        img = pan.augment_image(img)

    #zoom
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    #brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    #flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering


    return img,steering

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(320,160))
    img = img / 255
    return img

#imgRe= preProcessing(mpimg.imread('test.jpg'))
#plt.imshow(imgRe)
#plt.show()

def batchCreate(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            imf = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield (np.asarray(imgBatch),np.asarray(steeringBatch))


def createModel():
    model = Sequential()
    
    model.add(Convolution2D(24,(5,5),(2,2),input_shape= (320,160,3), activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(adam_v2.Adam(learning_rate = 0.0001), loss='mse')

    return model
