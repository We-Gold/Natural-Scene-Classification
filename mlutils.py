import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import os
import random

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

# Testing machine learning model
"""
def evaluate_multiclass_model(path_to_model,path_to_testing_data,target_image_size):
    model = load_model(path_to_model)

    total_correct = 0
    total_incorrect = 0

    for i in range(count_images(path_to_testing_data)):
        classes = os.listdir(path_to_testing_data)
        choice = random.randint(0,len(classes)-1)
        images = os.path.join(path_to_testing_data,classes[choice])
        image = os.listdir(images)[random.randint(0,len(images)-1)]
        image_path = os.path.join(images,image)
        if(classes[np.argmax(model.predict(image_to_tensor(image_path,target_image_size)))] == classes[choice]):
            total_correct+=1
        else:
            total_incorrect+=1
    return total_correct/(total_incorrect+total_correct)
"""
def evaluate_multiclass_model(path_to_model,path_to_testing_data,target_image_size):
    model = load_model(path_to_model)

    classes = os.listdir(path_to_testing_data)

    total_correct = 0
    total_incorrect = 0

    for i in os.listdir(path_to_testing_data):
        for image in os.listdir(os.path.join(path_to_testing_data,i)):
            image_path = os.path.join(path_to_testing_data,i,image)
            if(classes[np.argmax(model.predict(image_to_tensor(image_path,target_image_size)))] == i):
                total_correct+=1
            else:
                total_incorrect+=1
    return total_correct/(total_incorrect+total_correct)

def evaluate_multiclass_models(paths,path_to_testing_data,target_image_size):
    """Evaluates the accuracy of each model."""
    for path in paths:
        print(path[:-3].split("/")[-1]+": "+str(evaluate_multiclass_model(path,path_to_testing_data,target_image_size)))

def predict_image(classes,image_path,model_path,target_image_size):
    model = load_model(model_path)
    if(len(classes)>0):
        return classes[np.argmax(model.predict(image_to_tensor(image_path,target_image_size)))]
    else:
        return model.predict(image_to_tensor(image_path,target_image_size))


# Visualization of the dataset
