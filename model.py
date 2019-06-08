import os
import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

            
#set batch size
batch_size = 128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
# trim image to only see section with road
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)
history_object = model.fit_generator(train_generator, 
                 steps_per_epoch = math.ceil(len(train_samples)/batch_size), 
                 validation_data = validation_generator, 
                 validation_steps = math.ceil(len(validation_samples)/batch_size), 
                 epochs = 10, verbose = 1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.savefig('/home/workspace/CarND-Behavioral-Cloning-P3/loss.png')
plt.show()
exit()