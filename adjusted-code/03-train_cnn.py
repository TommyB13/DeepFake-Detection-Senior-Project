import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
from datetime import datetime

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)

# Set GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set to force CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Dataset paths
dataset_path = '../split_dataset'
tmp_debug_path = './tmp_debug'
os.makedirs(tmp_debug_path, exist_ok=True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

input_size = 128
batch_size_num = 16
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(rescale=1/255, rotation_range=10, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.2, zoom_range=0.1,
                                   horizontal_flip=True, fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_path, target_size=(input_size, input_size),
                                                    class_mode="binary", batch_size=batch_size_num, shuffle=True)

val_datagen = ImageDataGenerator(rescale=1/255)
val_generator = val_datagen.flow_from_directory(val_path, target_size=(input_size, input_size),
                                                class_mode="binary", batch_size=batch_size_num, shuffle=True)

test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(test_path, classes=['real', 'fake'], target_size=(input_size, input_size),
                                                  class_mode=None, batch_size=1, shuffle=False)

# Model definition
efficient_net = EfficientNetB0(weights='imagenet', input_shape=(input_size, input_size, 3),
                              include_top=False, pooling='max')
model = Sequential([
    efficient_net,
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint_filepath = './tmp_checkpoint'
os.makedirs(checkpoint_filepath, exist_ok=True)
custom_callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1),
    ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'best_model.keras'),
                    monitor='val_loss', mode='min', verbose=1, save_best_only=True)
]

# Train model
history = model.fit(train_generator, epochs=100, steps_per_epoch=len(train_generator),
                    validation_data=val_generator, validation_steps=len(val_generator), callbacks=custom_callbacks)

# Load best model and predict
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))
test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})

# Save results with datetime
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_path = os.path.join(results_dir, f"test_results_{current_time}.txt")

# Output DataFrame to file
test_results.to_csv(file_path, sep='\t', index=False)

print(f'Test results saved to {file_path}')

