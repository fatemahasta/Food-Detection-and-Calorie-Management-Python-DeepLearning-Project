from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from keras.models import load_model


train_path = '/home/fatema/Downloads/Python_Project_Food/train'
validation_path = '/home/fatema/Downloads/Python_Project_Food/validation'
test_path = '/home/fatema/Downloads/Python_Project_Food/test/'
classes = ['apple_pie', 'baklava', 'bread_pudding', 'carrot_cake', 'cheesecake', 'chicken_curry', 'chicken_wings', 'chocolate_cake', 'club_sandwich', 'cup_cakes']

######## Set up Image Augmentation
print("Setting up ImageDataGenerator")
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)=1/8
    height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)=1/8
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False, # randomly flip images
    rescale=1./255,
    fill_mode='nearest')

generator = datagen.flow_from_directory(train_path,target_size=(299,299),classes=classes, batch_size=10)
val_generator = datagen.flow_from_directory(validation_path,target_size=(299,299),classes=classes, batch_size=5)
test_generator = datagen.flow_from_directory(test_path,target_size=(299,299),class_mode=None,shuffle=False,batch_size=20)
model = load_model('second.3.07-0.85.hdf5')

#score = model.evaluate_generator(val_generator,500)
#print(score)

test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1,steps=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (generator.class_indices)
print(labels)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions)
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results1.csv",index=False)
