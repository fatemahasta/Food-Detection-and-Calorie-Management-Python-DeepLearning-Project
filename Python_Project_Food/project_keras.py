from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import keras.backend as K
from keras.optimizers import SGD, RMSprop, Adam
from time import time



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
    width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False, # randomly flip images
    rescale=1./255,
    fill_mode='nearest')

generator = datagen.flow_from_directory(train_path,target_size=(299,299),classes=classes, batch_size=10)
val_generator = datagen.flow_from_directory(validation_path,target_size=(299,299),classes=classes, batch_size=5)
test_generator = datagen.flow_from_directory(test_path,target_size=(299,299),class_mode=None,shuffle=False,batch_size=20)

K.clear_session()

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
# # x = Flatten()(x)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(.5)(x)
predictions = Dense(10, activation='softmax')(x)


model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='RMSprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("First pass")
checkpointer = ModelCheckpoint(filepath='/home/fatema/Downloads/Re%3a_Regarding_Python%2fDeep_learning_Project/first.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('first.3.log')
tensorboard=TensorBoard(log_dir="/home/fatema/Downloads/Re%3a_Regarding_Python%2fDeep_learning_Project/logs/{}".format(time())) #tensorboard declaration to visualize plot
model.fit_generator(generator,
                    validation_data=val_generator,
                    validation_steps=500,
                    steps_per_epoch=750,
                    nb_epoch=5,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer, tensorboard])

for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

print("Second pass")
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='/home/fatema/Downloads/Re%3a_Regarding_Python%2fDeep_learning_Project/second.3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('second.3.log')
model.fit_generator(generator,
                    validation_data=val_generator,
                    validation_steps=500, steps_per_epoch=750,
                    nb_epoch=10,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer, tensorboard]) #returns history object that has all the loss values.

model.save('model.h5')