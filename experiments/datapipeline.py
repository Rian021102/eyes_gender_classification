import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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