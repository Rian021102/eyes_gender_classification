from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
from tensorflow.keras import layers
from tensorflow import keras

def load_and_preprocess_images(file_paths, labels, image_size=50):
    data = []
    for path, label in zip(file_paths, labels):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(image_size, image_size), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        data.append([img_array, label])
    return np.array([item[0] for item in data]), np.array([item[1] for item in data])

def load_data_and_preprocess(data_path):
    path_img = list(data_path.glob('**/*.jpg'))
    labels = [os.path.split(path.parent)[1] for path in path_img]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    train_data = pd.DataFrame({
        'File_Path': path_img,
        'Labels': encoded_labels
    })

    train_data = train_data.sample(frac=1).reset_index(drop=True)

    X, y = load_and_preprocess_images(train_data['File_Path'], train_data['Labels'])

    return X, y

def resize_img(img, shape=(50, 50)):
    img = cv2.resize(img, shape)
    img = np.array(img)
    return img

def split_and_normalize_data(X, y, test_size=0.2):
    train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=test_size, random_state=42)

    train_images = np.array([resize_img(img) for img in train_images]) / 255.
    train_images = np.reshape(train_images, (len(train_images), 50, 50, 1))

    test_images = np.array([resize_img(img) for img in test_images]) / 255.
    test_images = np.reshape(test_images, (len(test_images), 50, 50, 1))

    return train_images, test_images, train_labels, test_labels

def build_model(learning_rate=0.001, dropout_rate=0.2, conv1_filters=32, conv2_filters=32):
    with tf.device('/cpu:0'):
        # use optimizer='adam' for training
        adam_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model = Sequential([
            layers.Conv2D(conv1_filters, 3, activation='relu', input_shape=(50, 50, 1)),
            layers.Conv2D(conv2_filters, 3, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid', name='gender')
        ])
        
        model.compile(
            optimizer=adam_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

def train_model(model, train_images, train_labels, test_images, test_labels, epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath='/Users/rianrachmanto/miniforge3/project/eyesgender/model/trained_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    train_images_gen = ImageDataGenerator(rotation_range=90, zoom_range=0.3, width_shift_range=0.3)
    train_images_aug = train_images_gen.flow(x=train_images, y=train_labels, batch_size=64)

    history = model.fit(
        train_images_aug,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        callbacks=[early_stopping, model_checkpoint]
    )

    return history

def main():
    data_path = Path('/Users/rianrachmanto/miniforge3/project/eyesgender/data/eyesfiles')
    X, y = load_data_and_preprocess(data_path)
    train_images, test_images, train_labels, test_labels = split_and_normalize_data(X, y)
    
    # Pass hyperparameters to build_model function
    model = build_model(learning_rate=0.0001, dropout_rate=0.5, conv1_filters=128, conv2_filters=256)
    
    history = train_model(model, train_images, train_labels, test_images, test_labels)
    # No need to save hyperparameters as json

if __name__ == "__main__":
    main()
