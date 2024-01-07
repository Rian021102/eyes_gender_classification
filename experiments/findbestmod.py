import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras import layers


def build_model_find(learning_rate=0.00001):
    with tf.device('/cpu:0'):
        # use optimizer='adam' for training
        adam_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        CNN=keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(64,64,3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3)),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(1024,activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1024,activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1,activation='sigmoid')  
        ])
        CNN.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return CNN
def train_model_find(CNN, X_train, y_train, X_test, y_test):
    model_checkpoint = ModelCheckpoint(
        filepath='/Users/rianrachmanto/miniforge3/project/eyesgender/model/CNN_find1.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    earlystopping=EarlyStopping(monitor='val_loss', patience=10,
                                verbose=1, mode='min', restore_best_weights=True)
    CNN_history = CNN.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
                          verbose=1,callbacks=[model_checkpoint,earlystopping])
    
    return CNN_history

def evaluate_model_find(CNN_history, CNN, X_test, y_test):
    # plot confusion matrix
    y_pred = CNN.predict(X_test)
    y_pred = np.round(y_pred)
    #print classification report
    print(classification_report(y_test, y_pred))
    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot the loss function and train/validation accuracies
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(CNN_history.history['loss'], label='train_loss')
    plt.plot(CNN_history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(CNN_history.history['accuracy'], label='train_accuracy')
    plt.plot(CNN_history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()