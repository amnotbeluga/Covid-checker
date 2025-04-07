import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random

# Define paths
original_covid_dir = r'C:\Users\ayush\PycharmProjects\COVID-19_Checker\data\covid'
original_normal_dir = r'C:\Users\ayush\PycharmProjects\COVID-19_Checker\data\normal'
train_dir = r'C:\Users\ayush\PycharmProjects\COVID-19_Checker\data\train'
val_dir = r'C:\Users\ayush\PycharmProjects\COVID-19_Checker\data\val'

train_covid_dir = os.path.join(train_dir, 'covid')
val_covid_dir = os.path.join(val_dir, 'covid')
train_normal_dir = os.path.join(train_dir, 'normal')
val_normal_dir = os.path.join(val_dir, 'normal')

# Create directories if they don't exist
os.makedirs(original_covid_dir, exist_ok=True)
os.makedirs(original_normal_dir, exist_ok=True)
os.makedirs(train_covid_dir, exist_ok=True)
os.makedirs(val_covid_dir, exist_ok=True)
os.makedirs(train_normal_dir, exist_ok=True)
os.makedirs(val_normal_dir, exist_ok=True)

# Function to split data into train and validation sets
def split_data(src_dir, train_dir, val_dir, split_ratio=0.8):
    filenames = os.listdir(src_dir)
    random.shuffle(filenames)
    train_size = int(len(filenames) * split_ratio)
    train_files = filenames[:train_size]
    val_files = filenames[train_size:]

    for file in train_files:
        if not os.path.exists(os.path.join(train_dir, file)):
            shutil.copy(os.path.join(src_dir, file), os.path.join(train_dir, file))

    for file in val_files:
        if not os.path.exists(os.path.join(val_dir, file)):
            shutil.copy(os.path.join(src_dir, file), os.path.join(val_dir, file))

# Split COVID-19 images
split_data(original_covid_dir, train_covid_dir, val_covid_dir)

# Split Normal images
split_data(original_normal_dir, train_normal_dir, val_normal_dir)

# Check directory structures
print(f"Train COVID directory: {os.listdir(train_covid_dir)[:5]}")
print(f"Train Normal directory: {os.listdir(train_normal_dir)[:5]}")
print(f"Val COVID directory: {os.listdir(val_covid_dir)[:5]}")
print(f"Val Normal directory: {os.listdir(val_normal_dir)[:5]}")

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (COVID-19 positive/negative)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=25, validation_data=val_generator)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Plot training & validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(25)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the model
model.save('covid19_detector_model.h5')
