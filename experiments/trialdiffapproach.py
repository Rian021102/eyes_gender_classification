import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras import layers
import visualkeras
import os
import warnings
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
warnings.filterwarnings('ignore')

def load_data(dir):
    label = []
    path = []
    for dirname, _, filenames in os.walk(dir):
        for filename in filenames:
            # Get the last part of the directory path as the label
            current_label = os.path.basename(dirname)
            
            # Skip 'eyesfiles' label
            if current_label != 'eyesfiles':
                label.append(current_label)
                path.append(os.path.join(dirname, filename))
    df = pd.DataFrame(columns=['Image', 'Label'])
    df['Image'] = path
    df['Label'] = label
    df = shuffle(df)
    df = df.reset_index(drop=True)
    return df

def resize_image_preprocess(df):
    size = (64,64)
    df1 = df.copy()
    for i in range(len(df)):
        image=cv2.imread(df['Image'][i])
        image=cv2.resize(image,size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        df1['Image'][i] = image
    LE = LabelEncoder()
    df1['Label'] = np.asarray(LE.fit_transform(df1["Label"]))
    return df1

def cleaned(X):
    for i in range(len(X)):
        X[i] = np.stack(X[i].reset_index(drop=True))
    return X

def to_tensor(_list):
    LIST = []
    for i in range(len(_list)):
        LIST.append(tf.convert_to_tensor(_list[i]))
    return LIST

def split_data(df1):
    # Split data into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(df1['Image'],df1['Label'],test_size=0.2,random_state=42)

    #cleaned X_train and X_test
    X_train, X_test,y_train,y_test = cleaned([X_train, X_test,y_train,y_test])

    # Convert to tensors
    X_train, X_test,y_train,y_test = to_tensor([X_train, X_test,y_train,y_test])
    return X_train, X_test,y_train,y_test

def build_model():
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
    return CNN
def train_model(CNN, X_train, y_train, X_test, y_test):
    CNN_history = CNN.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
                          verbose=1)
    
    # Save model
    CNN.save('CNN.h5')
    
    return CNN_history

def evaluate_model(CNN_history, CNN, X_test, y_test):
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

    


def main():
    dir = '/Users/rianrachmanto/miniforge3/project/eyesgender/data/eyesfiles'
    df = load_data(dir)
    # Print unique labels after loading the data
    print(df.Label.unique())
    df1 = resize_image_preprocess(df)
    print(df1.head())
    X_train, X_test, y_train, y_test = split_data(df1)
    CNN = build_model()
    print(CNN.summary())
    visualkeras.layered_view(CNN,to_file='output.png')
    CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    CNN_history = train_model(CNN, X_train, y_train, X_test, y_test)
    evaluate_model(CNN_history,CNN,X_test, y_test)



if __name__ == "__main__":
    main()
