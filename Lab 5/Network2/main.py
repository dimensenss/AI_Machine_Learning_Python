# import numpy as np
# from tensorflow import keras
# import random
#
# # # Завантаження Fashion MNIST
# # fashion_mnist = keras.datasets.fashion_mnist
# # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# #
# # # Нормалізація даних
# # train_images = train_images / 255.0
# # test_images = test_images / 255.0
# #
# # # Перетворення даних
# # train_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(train_images, train_labels)]
# # test_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(test_images, test_labels)]
#
# # Реалізація ReLU та її похідної
# def relu(z):
#     return np.maximum(0, z)
#
# def relu_prime(z):
#     return np.where(z > 0, 1, 0)
#
# class Network:
#
#     def __init__(self, sizes):
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
#         self.num_layers = len(self.sizes)
#
#     def feedforward(self, a):
#         for b, w in zip(self.biases, self.weights):
#             a = relu(np.dot(w, a) + b)
#         return a
#
#     def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
#         test_data = list(test_data)
#         n_test = len(test_data)
#         training_data = list(training_data)
#         n = len(training_data)
#         for j in range(epochs):
#             random.shuffle(training_data)
#             mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
#             for mini_batch in mini_batches:
#                 self.update_mini_batch(mini_batch, eta)
#             if test_data:
#                 print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
#             else:
#                 print(f"Epoch {j} complete")
#
#     def update_mini_batch(self, mini_batch, eta):
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         for x, y in mini_batch:
#             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#             nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#             nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#         self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
#         self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
#
#     def backprop(self, x, y):
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         activation = x
#         activations = [x]
#         zs = []
#         for b, w in zip(self.biases, self.weights):
#             z = np.dot(w, activation) + b
#             zs.append(z)
#             activation = relu(z)
#             activations.append(activation)
#         delta = self.cost_derivative(activations[-1], y) * relu_prime(zs[-1])
#         nabla_b[-1] = delta
#         nabla_w[-1] = np.dot(delta, activations[-2].transpose())
#         for l in range(2, self.num_layers):
#             z = zs[-l]
#             sp = relu_prime(z)
#             delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
#             nabla_b[-l] = delta
#             nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
#         return nabla_b, nabla_w
#
#     def evaluate(self, test_data):
#         test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
#         return sum(int(x == y) for (x, y) in test_results)
#
#     def cost_derivative(self, output_activations, y):
#         return output_activations - y
#
# # Векторизація міток
# def vectorized_result(j):
#     e = np.zeros((10, 1))
#     e[j] = 1.0
#     return e
#
# train_labels = [vectorized_result(label) for label in train_labels]
#
# # Функція для класифікації довільних зображень та візуалізації
# def classify_random_images(network, test_data, num_samples=2):
#     for _ in range(num_samples):
#         image_index = random.randint(0, len(test_data) - 1)
#         image_data, actual_label = test_data[image_index]
#         predicted_label = np.argmax(network.feedforward(image_data))
#         print("Predicted Label:", predicted_label)
#         print("Actual Label:", np.argmax(actual_label))
#         visualize_image(image_data)
#
# def visualize_image(image_data):
#     import matplotlib.pyplot as plt
#     image = np.reshape(image_data, (28, 28))
#     plt.imshow(image, cmap='gray')
#     plt.show()
#
# # Створення мережі
# net2 = Network([784, 40, 9])
#
# # Навчання мережі
# net2.SGD(train_data, epochs=1, mini_batch_size=9, eta=5.0, test_data=test_data)
#
# # Класифікація довільних зображень та візуалізація
# classify_random_images(net2, test_data)

import random  # бібліотека функцій для генерації випадкових значень
import numpy as np
from matplotlib import pyplot as plt


def relu(z): # визначення RELU функції активації
    return np.maximum(0, z)

def relu_prime(z):  # Похідна гладкого наближення RELU
    return 1.0/(1.0+np.exp(-z))


class Network(object):  # використовується для опису нейронної мережі

    def __init__(self, sizes):  # Конструктор класу
        self.num_layers = len(sizes)  # Визначаємо кількість шарів в мережі
        self.sizes = sizes  # Зберігаємо розміри кожного шару
        self.biases = [np.random.randn(y, 1) for y in
                       sizes[1:]]  # Ініціалізація випадкових зміщень для кожного нейрона, окрім вхідного шару
        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = relu(np.dot(w, a) + b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data):
        test_data = list(test_data)  # створюємо список об'єктів тестової вибірки
        n_test = len(test_data)  # обчислюємо довжину тестової вибірки
        training_data = list(training_data)  # створюємо список об'єктів навчальної вибірки
        n = len(training_data)  # обчислюємо розмір навчальної вибірки
        for j in range(epochs):  # цикл по эпохам
            random.shuffle(training_data)  # змішуємо елементи навчальної вибірки
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]  # створюємо підвибірки
            for mini_batch in mini_batches:  # цикл по подвибіркам
                self.update_mini_batch(mini_batch, eta)  # один крок градієнтного спуску
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))  # дивимось прогрес у навчанні

    def update_mini_batch(self  , mini_batch  , eta ):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # список градієнтів dC/db для кожного шару (спочатку заповнюється нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # список градієнтів dC/dw для кожного шару (спочатку заповнюється нулями)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # пошарово обчислюємо градієнти dC/db та dC/dw для поточного прикладу (x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  #підсумовуємо градієнти dC / db для різних прикладів поточної підвиборки
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  #підсумовуємо градієнти dC / dw для різних прикладів поточної підвиборки

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]  #оновлюємо всі ваги w нейронної мережі
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]  # оновлюємо всі зміщення b нейронної мережі



    def backprop(self, x , y ):
        nabla_b = [np.zeros(b.shape) for b in
                   self.biases]  # список градієнтів dC/db для кожного шару (спочатку заповнюється нулями)

        nabla_w = [np.zeros(w.shape) for w in
               self.weights]  # список градієнтів dC/dw для кожного шару (спочатку заповнюється нулями)

        # визначення змінних
        activation = x  # вихідні сигнали шару (спочатку відповідають  вихідним сигналам 1-го шару або вхідним сигналам мережі)
        activations = [x]  # список вихідних сигналів по всім шарам (спочатку містить тільки вихідні сигнали 1-го шару)
        zs = []  # список активаційних потенціалів по всім шарам (спочатку пустий)

        # пряме розповсюдження
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # зчитуємо активаційні потенціали поточного шару
            zs.append(z)  # додаємо елемент (активаційні потенціали шару) в кінець списку
            activation = relu(z)  # зчитуємо вихідні сигнали поточного шару, застосовуючи сигмоїдальну функцію активації до активаційних потенціалів шару
            activations.append(activation)  # додаємо елемент (вихідні сигнали шару) в кінець списку

        # зворотне розповсюдження
        delta = self.cost_derivative(activations[-1], y) * relu_prime(zs[-1])  # зчитуємо міру впливу нейронів вихідного шару L на величину помилки (BP1)
        nabla_b[-1] = delta  # градієнт dC/db для шару L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градієнт dC / dw для шару L(BP4)

        for l in range(2, self.num_layers):
            z = zs[-l]  # активаційні потенціали l-го шару (рухаємось по  списку справа наліво)
            sp = relu_prime(z)  # зчитуємо сигмоїдальну функцію від активаційних потенціалів l-го шару
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  #зчитуємо міру впливу нейронів l - го шару на величину помилки(BP2)
            nabla_b[-l] = delta  # градієнт dC/db для l-го шару (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #градієнт dC / dw для l - го шару(BP4)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):  # Оцінка прогресу в навчанні
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations,y):  # Обчислення приватних похідних функції вартості по вихідним сигналам останнього шару
        return (output_activations - y)


def image_selection(net):
    index1 = random.randint(0, len(test_data) - 1)
    index2 = random.randint(0, len(test_data) - 1)

    # Отримаємо вхідні дані та мітки за вибраними індексами
    image1, label1 = test_data[index1]
    image2, label2 = test_data[index2]

    # Класифікуємо вхідні дані з допомогою навченої мережі
    prediction1 = net.feedforward(image1)
    prediction2 = net.feedforward(image2)

    # Візуалізуємо результати
    visualize_classification_result(image1, label1, prediction1)
    visualize_classification_result(image2, label2, prediction2)

def visualize_classification_result(image, label, prediction):
    # Розміщення зображення
    plt.subplot(1, 2, 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {label}")

    # Виведемо результат класифікації
    plt.subplot(1, 2, 2)

    # Розділимо значення prediction на окремі класи
    classes = list(range(10))
    plt.bar(classes, prediction.flatten())  # Використовуємо .flatten() для перетворення на одновимірний масив
    plt.xticks(range(10), [str(i) for i in range(10)])  # Встановлюємо значення на горизонтальній осі як цілі числа
    plt.title("Network's Prediction")
    plt.show()


if __name__ == "__main__":
    # from Network2.fashion_mnist_loader import load_data_wrapper
    # training_data, test_data = load_data_wrapper()
    from Network1.mnist_loader import load_data_wrapper
    training_data, validation_data, test_data = load_data_wrapper()

    net = Network([784, 30, 10])
    net.SGD(training_data, 10, 16, 0.065, test_data=test_data)
    image_selection(net)




