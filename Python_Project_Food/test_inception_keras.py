#from keras import Sequential
#from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam

train_path = '/home/fatema/Downloads/Re%3a_Regarding_Python%2fDeep_learning_Project/train'
validation_path = '/home/fatema/Downloads/Re%3a_Regarding_Python%2fDeep_learning_Project/validation'
classes = ['apple_pie', 'baklava', 'bread_pudding', 'carrot_cake', 'cheesecake', 'chicken_curry', 'chicken_wings', 'chocolate_cake', 'club_sandwich', 'cup_cakes']
train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(299,299),classes=classes,batch_size=10)
validation_batches = ImageDataGenerator().flow_from_directory(validation_path,target_size=(299,299),classes=classes,batch_size=5)
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

for layer in model.layers:
    layer.trainable = False
    
epochs = 20
lrate = 0.0001
decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(RMSprop(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

model.fit_generator(train_batches,validation_data= validation_batches, validation_steps=500, steps_per_epoch=750, epochs=epochs,verbose=2)