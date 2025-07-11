import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Set paths
train_dir = "data/raw/train"
test_dir = "data/raw/test"

# Image generators
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(48, 48),
                                           color_mode='grayscale', class_mode='categorical',
                                           batch_size=64)
test_data = test_gen.flow_from_directory(test_dir, target_size=(48, 48),
                                         color_mode='grayscale', class_mode='categorical',
                                         batch_size=64)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
checkpoint = ModelCheckpoint("model/emotion_model.h5", save_best_only=True)
model.fit(train_data, validation_data=test_data, epochs=20, callbacks=[checkpoint])
