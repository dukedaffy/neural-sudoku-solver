import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy





def predict(loc,mod):

    #load the saved model
    model = load_model(mod)

    model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    matrix=[]
    for i in range(81):
        #loading images from your system
        path=loc + str(i) +'.png' 
        img1 = cv2.imread(path)

        #converting the image to grayscale
        img= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        #converting to binary image
        #et, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 

        #resizing the image
        img = cv2.resize(img,(28,28))
        img = np.reshape(img,[1,28,28,1])
        img1 = np.reshape(img,[28,28,1])

        img1= img1[10:20, 10:20] 

        count=0
        for i in range(10):
            for j in range(10):
                if (img1[i][j][0]==0):
                        count+=1

        if(count>80):
            matrix.append(0)
        else:

            #predicting the model
            pre = model.predict(img)
            matrix.append(np.argmax(pre))
        

    arr = to_matrix(matrix,9)
    a=matrixflip(arr)
    return(a)

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def matrixflip(tempm):
    
    for i in range(0,len(tempm),1):
            tempm[i].reverse()
    
    tempm.reverse()
    tempm=np.transpose(tempm)
    return(tempm)



