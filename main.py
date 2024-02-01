import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse


def modelarchitecture():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    lr_schedule = ExponentialDecay(
            0.001,
            decay_steps=1500,
            decay_rate=0.8,
            staircase=True)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    

    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def main():

    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    train_x = train_x[:,:,:,np.newaxis] 
    test_x = test_x[:,:,:,np.newaxis]
    train_x = train_x/255
    test_x = test_x/255

    model = modelarchitecture()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=10, validation_split = 0.2, callbacks=[early_stopping])

    model.save('number_classifier.model')
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print(f"Test loss {test_loss}")
    print(f"Test accuracy {test_accuracy}")

    image = np.reshape(img, (1, 28, 28, 1))  # Assuming the model expects a single-channel image

    prediction = model.predict(image)
    index = np.argmax(prediction)
    print(index-1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image file')
    args = parser.parse_args()
    input_image = args.input_image

    img = cv.imread(input_image, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (28, 28))

    
    main()



