import numpy as np
from tensorflow import keras


def load_data_wrapper():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    #Змінення розмірності зображень у вхідні дані для навчання
    training_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in train_images]
    #Нормалізація значень пікселів до діапазону [0, 1]
    training_results = [vectorized_result(y) for y in train_labels]
    training_data = list(zip(training_inputs, training_results))

    # Перетворення набору даних тестування
    test_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in test_images]
    test_data = list(zip(test_inputs, test_labels))

    return training_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


