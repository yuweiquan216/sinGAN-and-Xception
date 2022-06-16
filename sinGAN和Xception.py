import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

image_path = '/kaggle/input/kermany2018/oct2017/OCT2017 '
oct_csv_path = '/kaggle/input/oct-csv/'
oct_singan_path = '/kaggle/input/octsingan/'
train_dir = image_path + "/train/"
valid_dir = image_path + "/val/"
test_dir = image_path + "/test/"

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
cols = [x.upper() for x in classes]
dirs = [train_dir, valid_dir, test_dir]
label = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
IMG_SIZE = 224

# if we should read the directory structre, if False then use the CSV files already saved
# Once you generate the csv files you should probably download them and re-upload into kaggle and set this to FALSE
REGEN = False 

def create_df (path, classes=classes):
  df = pd.DataFrame(columns=['FILENAME', 'CNV', 'DME', 'DRUSEN', 'NORMAL'])
  for sub_dir in classes:
    condition = {'NORMAL': 0, 'CNV': 0, 'DME':0, 'DRUSEN': 0}
    files = os.listdir(path + sub_dir)
    if (sub_dir== 'NORMAL'):
      condition['NORMAL'] = 1
    elif (sub_dir == 'CNV'):
      condition['CNV'] = 1
    elif (sub_dir == 'DME'):
      condition['DME'] = 1
    else:
      condition['DRUSEN']= 1
    for f in files:
      df = df.append({'FILENAME': path +  sub_dir  + "/" + f, 
                      'NORMAL': condition['NORMAL'], 
                      'CNV': condition['CNV'],
                      'DME': condition['DME'],
                      'DRUSEN': condition['DRUSEN']}, ignore_index=True)
  return df

# Generting the DataFrames of the filenames
# this is primarily used so we can sub-sample files easier for the different training strategies
if (REGEN):
  train_df = create_df(train_dir)
  valid_df = create_df(valid_dir)
  test_df = create_df(test_dir)
  singan_df = create_df(oct_singan_path)
  train_df.to_csv("train_data.csv")
  valid_df.to_csv("valid_data.csv")
  test_df.to_csv("test_data.csv")
  singan_df.to_csv("singan_data.csv")
else:
  train_df = pd.read_csv(oct_csv_path + "train_data.csv")
  valid_df = pd.read_csv(oct_csv_path + "valid_data.csv")
  test_df = pd.read_csv(oct_csv_path + "test_data.csv")
  singan_df = pd.read_csv(oct_csv_path + "singan_data.csv")

print ("Training Data: ", train_df.shape)
print ("Validation Data: ", valid_df.shape)
print ("Test Data: ", test_df.shape)
print ("SinGAN Data: ", singan_df.shape)

# Printing out the # of samples for each subsample percentage 
print ("Trainig Data percentages:")
print (" 1% ==> ", int(.01 * train_df.shape[0]))
print (" 5% ==> ", int(.05 * train_df.shape[0]))
print ("10% ==> ", int(.1  * train_df.shape[0]))
print ("25% ==> ", int(.25 * train_df.shape[0]))
print ("50% ==> ", int(.5  * train_df.shape[0]))
print ("75% ==> ", int(.75 * train_df.shape[0]))
print ("90% ==> ", int(.9  * train_df.shape[0]))
print ("98% ==> ", int(.98 * train_df.shape[0]))

singan_df.head(10)

# Sampling 50% of the data
sample = train_df.sample(frac=0.1, random_state=10, axis=0)
sample = sample.append(singan_df, ignore_index=False)
sample = sample.sample(frac=1, random_state=10, axis=0)

# determine class weights to feed into neural network during training
def get_classweight(df):
  total = df.shape[0]
  num_norm = df['NORMAL'].sum()
  num_cnv = df['CNV'].sum()
  num_dme = df['DME'].sum()
  num_drusen = df['DRUSEN'].sum()
  norm_weight = (1/num_norm) * (total/4)
  cnv_weight = (1/num_cnv) * (total/4)
  dme_weight = (1/num_dme) * (total/4)
  drusen_weight = (1/num_drusen) * (total/4)
  class_weight = {0 : cnv_weight, 1: dme_weight,
                           2 : drusen_weight, 3: norm_weight}
  return class_weight

class_weight = get_classweight(sample)
class_weight

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow.keras.applications as app
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_image_datagen = ImageDataGenerator(rotation_range=90, width_shift_range=[-.1,.1], height_shift_range=[-.1,.1],
                                         shear_range=0.25, zoom_range=0.3, horizontal_flip=True,
                                         vertical_flip=True, rescale = 1./255., validation_split=0.1)

# Setting the imgages to come from the dataframe where we specify the filenames and columns to use for "labels"
train_imgs = train_image_datagen.flow_from_dataframe(sample, directory=None, x_col='FILENAME', y_col=cols, subset="training",
                                        class_mode="raw", target_size=(IMG_SIZE,IMG_SIZE), batch_size=32, seed=10)
valid_imgs = train_image_datagen.flow_from_dataframe(sample, directory=None, x_col='FILENAME', y_col=cols, subset="validation",
                                        class_mode="raw", target_size=(IMG_SIZE,IMG_SIZE), batch_size=32, seed=10)

# Creating the model based on Xception Network
input_layer = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = app.xception.Xception(include_top=False, weights="imagenet", input_shape=(IMG_SIZE,IMG_SIZE,3))
base_model.trainable = True

x = base_model(input_layer)
x = keras.layers.GlobalAveragePooling2D()(x)
output = keras.layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs=input_layer, outputs=output)
model.summary()

# This code did not work, it caused I/O Error 5:
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics='accuracy')
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

# Creating a checkpoint to save the best model so that we can reload it once training is complete
checkpoint_cb = keras.callbacks.ModelCheckpoint("oct_singan.h5", save_best_only=True)
# Adding an an early stop callback to avoid overfitting in case the model is not improving after 5 consescutive epochs
earlystop_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(train_imgs,  epochs=30, verbose=1, validation_data=valid_imgs, 
                    class_weight=class_weight, callbacks=[checkpoint_cb, earlystop_cb])

test_image_datagen = ImageDataGenerator( rescale = 1./255.)

test_imgs = test_image_datagen.flow_from_dataframe(test_df, directory=None, x_col='FILENAME', y_col=cols, validate_filenames=True,
                                        class_mode="raw", target_size=(224,224), batch_size=32, shuffle=False)

model.load_weights("oct_singan.h5")
model.evaluate(test_imgs)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
plot_acc(history)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
plot_loss(history)

tf.saved_model.save(model, 'XceptionSinGANOCT')

# Intialize the TFLite converter to load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('XceptionSinGANOCT')# YOUR CODE HERE

# Set the optimization strategy for 'size' in the converter 
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] # YOUR CODE HERE]

# Use the tool to finally convert the model
tflite_model = converter.convert()

tflite_model_file = 'XceptionSinGANOCT.tflite'

with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)