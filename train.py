import keras
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNetV2
from keras.applications import ResNet101V2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from mlutils import *
import os
import math

(train_generator, validation_generator) = get_data_generators(os.path.join(os.getcwd(),"data"),(150,150),16,"categorical")

base_model = ResNet101V2(weights = 'imagenet',include_top = False) 

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dense(512,activation='relu')(x)
x = Dense(256,activation='relu')(x)
preds = Dense(6,activation = 'softmax')(x)

model = Model(inputs = base_model.input,outputs = preds)

model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

checkpoints = ModelCheckpoint("checkpoints/weights.{epoch:02d}.h5",
                                          save_weights_only = False,
                                          verbose = 1)

total_train = count_images(os.path.join(os.getcwd(),"data/train"))
total_test = count_images(os.path.join(os.getcwd(),"data/test"))

model.fit_generator(train_generator,
                        steps_per_epoch=math.floor(total_train)/8,
                        epochs=20,
                        validation_data=validation_generator,
                        validation_steps=math.floor(total_test)/8,
                        callbacks = [checkpoints])

model.save("res-model.h5")