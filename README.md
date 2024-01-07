# Sex Classfication Based on Eyes Using CNN

## Introduction

"Dari Mata Turun ke Hati" is a widely recognized Indonesian phrase, loosely translated into English as "falling in love begins from the sight," akin to the concept of love at first sight. In the realm of artificial intelligence (A.I.), leveraging the eyes as tools can offer various benefits, such as identity identification and enhanced security, especially in the context of COVID-19 protocols that necessitate mask-wearing. Since only the eyes are visible, employing A.I. for sex identification becomes pivotal.
This project aims to address the challenges posed by limited facial visibility and contribute a solution to sex classification using a deep learning model, specifically Convolutional Neural Network (CNN). Beyond individual identification, the proposed system can find applications in businesses, aiding in sex-based customer segmentation. For instance, it can be utilized in stores to analyze the demographic distribution of visitors, providing valuable insights for business optimization.

## Dataset

The dataset employed in this project is sourced from Kaggle and is accessible here [https://www.kaggle.com/datasets/pavelbiz/eyes-rtte]. It comprises two folders: "femaleeyes" and "maleeyes." To utilize the dataset, download it and copy the folders into your local directory. It's important to note that the dataset lacks explicit labels. However, a labeling method will be employed, associating each image with its respective category based on the directory structure.
``````
def load_and_preprocess_images(file_paths, labels, image_size=100):
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

``````
## Binary Classification
The aim of this project is sex classification using eyes, hence there are only two expected output, female or male hence it makes this project as binary classfication. Since it's binary clasfication, the node activation used is Sigmoid where is defined as
![Alt text](image.png)


