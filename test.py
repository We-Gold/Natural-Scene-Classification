import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import random
import numpy as np
from mlutils import *

models = []
path = os.getcwd()
models.append(path+"/res-model.h5")
models.append(path+"/mobile2-model.h5")
models.append(path+"/custom-model2.h5")

evaluate_multiclass_models(models,os.path.join(os.getcwd(),"data/test"),(150,150))
#print(evaluate_multiclass_model(models[0],os.path.join(os.getcwd(),"data/test"),(150,150)))

"""
model = load_model(models[0])

total_correct = 0
total_incorrect = 0

count = 0

while count < count_images(os.path.join(os.getcwd(),"data/test")):
    classes = os.listdir(os.path.join(os.getcwd(),"data/test"))
    choice = random.randint(0,len(classes)-1)
    images = os.path.join(os.getcwd(),"data/test",classes[choice])
    image = os.listdir(images)[random.randint(0,len(images)-1)]
    image_path = os.path.join(images,image)
    if(classes[np.argmax(model.predict(image_to_tensor(image_path,(150,150))))] == classes[choice]):
        total_correct+=1
    else:
        total_incorrect+=1
    count+=1

print("Accuracy: "+ str(total_correct/(total_correct+total_incorrect)))
    
"""