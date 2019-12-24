import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Data generators, total images in folders, image to tensor
def get_data_generators(directory,target_image_size,batch_size,class_mode,zoom=0.2,horizontal_flip=True):
    """Returns a tuple containing the training and validation generators."""
    train_datagen = ImageDataGenerator(rescale=1./255,zoom_range=zoom,horizontal_flip=horizontal_flip)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(os.path.join(directory,"train"),target_size=target_image_size,batch_size=batch_size,class_mode=class_mode)
    validation_generator = test_datagen.flow_from_directory(os.path.join(directory,"test"),target_size=target_image_size,batch_size=batch_size,class_mode=class_mode)
    return (train_generator,validation_generator)

def image_to_tensor(img_path,target_image_size):
    """"Converts the specified image to a tensor that can be fed into a network."""
    img = image.load_img(img_path, target_size=target_image_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    return img_tensor

def count_files(directory):
    return len(os.listdir(directory))

def count_images(directory):
    """Count the number of images in all classes in either the training or testing folder."""
    total = 0
    for folder in os.listdir(directory):
        total += count_files(os.path.join(directory,folder))
    return total


