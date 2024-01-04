import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
import cv2
import json
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path, image_size=64):
    img = image.load_img(file_path, target_size=(image_size, image_size), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = cv2.resize(img_array, (image_size, image_size))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def load_hyperparameters(file_path='/Users/rianrachmanto/miniforge3/project/eyesgender/model/hyperparameters.json'):  # Update with the path to your hyperparameters file
    with open(file_path, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters

def load_trained_model(model_path='/Users/rianrachmanto/miniforge3/project/eyesgender/model/trained_model.h5'):  # Update with the path to your trained model file
    return tf.keras.models.load_model(model_path)

def make_prediction(model, image_array):
    prediction = model.predict(image_array)
    return prediction

def display_image_with_prediction(image_array, prediction):
    plt.imshow(np.squeeze(image_array), cmap='gray')  # Display the grayscale image
    plt.title(f"Prediction: {['Female', 'Male'][int(round(prediction[0][0]))]} with probability {prediction[0][0]:.2f}")
    plt.show()

def main():
    data_path = Path('/Users/rianrachmanto/miniforge3/project/eyesgender/data/depositphotos_79817084-stock-photo-male-eyes.jpg')  # Update with the path to the image you want to predict
    image_array = load_and_preprocess_image(str(data_path))

    # Load hyperparameters and trained model
    hyperparameters = load_hyperparameters()
    model = load_trained_model()

    # Make prediction
    prediction = make_prediction(model, image_array)

    # Display the image with prediction
    display_image_with_prediction(image_array, prediction)

if __name__ == "__main__":
    main()
