"""
mnist_loader.py
~~~~~~~~~~
Модуль для підключення та використання бази даних MNIST.

Група:[Вказати номер групи] ПІБ:[Вказати ПІБ студента] """

import gzip # бібліотека для стиснення та розпакування файлів gzip та gunzip.
import pickle # бібліотека для збереження та завантаження складних об'єктів  Python
import numpy as np # бібліотека для роботи з матрицями

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb') # відкриваємо стиснений файл gzip у двійковому режимі
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1') # завантажуємо таблиці з файлу
    f.close() # закриваємо файл
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data() # ініціалізація наборів даних у форматі MNIST
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] #перетворення масивів розміру 1 на 784 до масивім розмеру 784 на 1
    training_results = [vectorized_result(y) for y in tr_d[1]] #представлення цифр від 0 до 9 у вигляді масивів розміру 10 на 1
    training_data = zip(training_inputs, training_results) # формуємо набір навчальних даних з пар (x, y)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]] #перетворення масивів розміру 1 на 784 до масивів розміру 784 на 1
    validation_data = zip(validation_inputs, va_d[1]) # формуємо набір даних для перевірки з пар (x, y)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]] #перетворення масивів розміру 1 на 784 до масивів розміру 84 на 1
    test_data = zip(test_inputs, te_d[1]) # формуємо набір тестових данихз пар (x, y)
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


