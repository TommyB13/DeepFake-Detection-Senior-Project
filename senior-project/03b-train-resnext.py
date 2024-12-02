import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from timm import create_model

input_size = 128
batch_size_num = 32
train_path = './split_dataset/train'
val_path = './split_dataset/val'
test_path = './split_dataset/test'

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1/255,    # Rescale the tensor values to [0, 1]
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

val_datagen = ImageDataGenerator(
    rescale=1/255
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

test_datagen = ImageDataGenerator(
    rescale=1/255
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# Train a CNN classifier using ResNeXt
resnext = create_model(
    'resnext50_32x4d',
    pretrained=True,
    num_classes=0,
    in_chans=3,
    global_pool='avg'
)

resnext.trainable = True

# Wrap the model in a Sequential model
model = Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(input_size, input_size, 3)))
model.add(resnext)
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './tmp_checkpoint'
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'best_model.keras'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

# Train network
num_epochs = 100
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)
print(history.history)

# Load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))

# Generate predictions
test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose=1
)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
print(test_results)

