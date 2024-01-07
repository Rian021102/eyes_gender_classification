from datapipeline import load_data, resize_image_preprocess, split_data
from model_base import build_model_base, train_model_base, evaluate_model_base
from findbestmod import build_model_find, train_model_find, evaluate_model_find
import visualkeras


def main():
    dir = '/Users/rianrachmanto/miniforge3/project/eyesgender/data/eyesfiles'
    df = load_data(dir)
    # Print unique labels after loading the data
    print(df.Label.unique())
    df1 = resize_image_preprocess(df)
    print(df1.head())
    X_train, X_test, y_train, y_test = split_data(df1)
    CNN = build_model_find()
    print(CNN.summary())
    visualkeras.layered_view(CNN,to_file='output.png')
    #CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    CNN_history = train_model_find(CNN, X_train, y_train, X_test, y_test)
    evaluate_model_find(CNN_history,CNN,X_test, y_test)

if __name__ == "__main__":
    main()
