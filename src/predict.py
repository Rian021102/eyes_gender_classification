import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def load_and_preprocess_image(file_path, image_size=50):
    img = image.load_img(file_path, target_size=(image_size, image_size), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = cv2.resize(img_array, (image_size, image_size))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def load_trained_model(model_path='/Users/rianrachmanto/miniforge3/project/eyesgender/model/trained_model.h5'):  # Update with the path to your trained model file
    return tf.keras.models.load_model(model_path)

def make_prediction(model, image_array):
    prediction = model.predict(image_array)
    return prediction

def display_image_with_prediction(image_array, prediction, threshold=0.4):
    gender_label = 'Male' if prediction[0][0] >= threshold else 'Female'
    plt.imshow(np.squeeze(image_array), cmap='gray')  # Display the grayscale image
    plt.title(f"Prediction: {gender_label} with probability {prediction[0][0]:.2f}")
    plt.show()

def main():
    data_path = Path('/Users/rianrachmanto/miniforge3/project/eyesgender/data/360_F_55092987_uWxxwrInmIaPA68uE8ntECds4Fg28pls.jpg')  # Update with the path to the image you want to predict
    image_array = load_and_preprocess_image(str(data_path))

    # Load trained model
    model = load_trained_model()

    # Make prediction
    prediction = make_prediction(model, image_array)

    # Display the image with prediction (threshold set to 0.4)
    display_image_with_prediction(image_array, prediction, threshold=0.4)

if __name__ == "__main__":
    main()
