import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle
# from keras.utils.np_utils import to_categorical
##############################################
path = "dataset"
images = []
classNo = []
test_ratio = 0.2
validation_ratio = 0.2
imageDimensions = (32,32,3)
batchSizeVal = 50
epochsVal = 10
# stepsPerEpochval = len(X_train)// batchSizeVal

##############################################
myList = os.listdir(path)
print("Total No. classes",(len(myList)))

noOfClasses = len(myList)

print("Importing classes")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, (imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end = " ")
print(" ")
images = np.array(images)
classNo = np.array(classNo)

print("total data",images.shape)
# print(classNo.shape)

####spliting data
X_train, X_test, y_train, y_test = train_test_split(images,classNo,test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size= validation_ratio)
   
print("training data",X_train.shape)       
print("testing data",X_test.shape)
print("validation data",X_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append((len(np.where(y_train == x)[0])))
    
print(numOfSamples)


#looking for the data distribution for each class
plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No. of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

#preproccessing the data
def preproccessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


# img = X_train[32]
# img = cv2.resize(img,(300,300))  #why resizeing the image ? earlier it was 300*300 we resized it to 32*32 and now we are resizing it back to 300*300 for better visualization why?
# cv2.imshow("preprocessed image", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preproccessing,X_train)))
X_test = np.array(list(map(preproccessing,X_test)))
X_validation = np.array(list(map(preproccessing,X_validation)))



X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFliter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500
    
    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFliter1,input_shape=(imageDimensions[0],
                                                              imageDimensions[1],
                                                              1), activation = "relu")))
    model.add((Conv2D(noOfFilters, sizeOfFliter1,activation = "relu")))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation = "relu")))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation = "relu")))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

#variable for stepsPerEpoch
stepsPerEpochval = len(X_train)// batchSizeVal


history = model.fit(dataGen.flow(X_train,y_train,batch_size= batchSizeVal),
                                                            steps_per_epoch= stepsPerEpochval, 
                                                            epochs = epochsVal, 
                                                            validation_data = (X_validation,y_validation),
                                                            shuffle = 1)
    
    
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score',score[0])
print('Test Accuracy =', score[1])

pickle_out= open("trained_model/model_trained.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()