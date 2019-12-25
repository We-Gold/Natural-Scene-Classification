import keras
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Conv2D,MaxPool2D,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from mlutils import *
import os
import math

(train_generator, validation_generator) = get_data_generators(os.path.join(os.getcwd(),"data"),(150,150),16,"categorical")

model = Sequential()

model.add(Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Dropout(rate=0.5))
model.add(MaxPool2D(5,5))
model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(120,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(5,5))
model.add(Flatten())
model.add(Dense(180,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(6,activation='softmax'))

model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

#model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

checkpoints = ModelCheckpoint("checkpoints/weights.{epoch:02d}.h5",
                                          save_weights_only = False,
                                          verbose = 1)

total_train = count_images(os.path.join(os.getcwd(),"data/train"))
total_test = count_images(os.path.join(os.getcwd(),"data/test"))

model.fit_generator(train_generator,
                        steps_per_epoch=math.floor(total_train/16),
                        epochs=20,
                        validation_data=validation_generator,
                        validation_steps=math.floor(total_test/16),
                        callbacks = [checkpoints])

model.save("custom-model2.h5")