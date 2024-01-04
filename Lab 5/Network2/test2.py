# import random
# import numpy as np
#
# def relu(z):
#     return np.maximum(0.01*z, z)
#
# def relu_prime(z):
#     return np.where(z > 0, 1, 0)
#
# class Network:
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x) for x, y in zip(sizes[:-1], sizes[1:])]
#
#     def feedforward(self, a):
#         for b, w in zip(self.biases, self.weights):
#             a = relu(np.dot(w, a) + b)
#         return a
#
#     def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
#         test_data = list(test_data)
#         n_test = len(test_data)
#         training_data = list(training_data)
#         n = len(training_data)
#         for j in range(epochs):
#             random.shuffle(training_data)
#             mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
#             for mini_batch in mini_batches:
#                 self.update_mini_batch(mini_batch, eta)
#             accuracy = self.evaluate(test_data) / n_test
#             print(f"Epoch {j}: Accuracy {accuracy:.2%}")
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
#     def backprop(self, x , y ):
#         nabla_b = [np.zeros(b.shape) for b in
#                    self.biases]  # список градієнтів dC/db для кожного шару (спочатку заповнюється нулями)
#
#         nabla_w = [np.zeros(w.shape) for w in
#                self.weights]  # список градієнтів dC/dw для кожного шару (спочатку заповнюється нулями)
#
#         # визначення змінних
#         activation = x  # вихідні сигнали шару (спочатку відповідають  вихідним сигналам 1-го шару або вхідним сигналам мережі)
#         activations = [x]  # список вихідних сигналів по всім шарам (спочатку містить тільки вихідні сигнали 1-го шару)
#         zs = []  # список активаційних потенціалів по всім шарам (спочатку пустий)
#
#         # пряме розповсюдження
#         for b, w in zip(self.biases, self.weights):
#             z = np.dot(w, activation) + b  # зчитуємо активаційні потенціали поточного шару
#             zs.append(z)  # додаємо елемент (активаційні потенціали шару) в кінець списку
#             activation = relu(z)  # зчитуємо вихідні сигнали поточного шару, застосовуючи сигмоїдальну функцію активації до активаційних потенціалів шару
#             activations.append(activation)  # додаємо елемент (вихідні сигнали шару) в кінець списку
#
#         # зворотне розповсюдження
#         delta = self.cost_derivative(activations[-1], y) * relu_prime(zs[-1])  # зчитуємо міру впливу нейронів вихідного шару L на величину помилки (BP1)
#         nabla_b[-1] = delta  # градієнт dC/db для шару L (BP3)
#         nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градієнт dC / dw для шару L(BP4)
#
#         for l in range(2, self.num_layers):
#             z = zs[-l]  # активаційні потенціали l-го шару (рухаємось по  списку справа наліво)
#             sp = relu_prime(z)  # зчитуємо сигмоїдальну функцію від активаційних потенціалів l-го шару
#             delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  #зчитуємо міру впливу нейронів l - го шару на величину помилки(BP2)
#             nabla_b[-l] = delta  # градієнт dC/db для l-го шару (BP3)
#             nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #градієнт dC / dw для l - го шару(BP4)
#         return (nabla_b, nabla_w)
#
#     def evaluate(self, test_data):
#         test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
#         return sum(int(x == y) for (x, y) in test_results)
#
#     def cost_derivative(self, output_activations, y):
#         return (output_activations - y)
#
# # Остальной код оставьте без изменений
#
# if __name__ == "__main__":
#     from Network2.fashion_mnist_loader import load_data_wrapper
#     training_data, test_data = load_data_wrapper()
#     net = Network([784, 30, 10])
#     net.SGD(training_data, 30, 64, 0.001, test_data=test_data)

import numpy as np
from tensorflow import keras


# Загрузка данных и их предобработка
def load_data_wrapper():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # Перетворення набору даних навчання
    training_inputs = train_images.reshape(train_images.shape[0], 784) / 255.0
    training_results = keras.utils.to_categorical(train_labels, 10)
    training_data = list(zip(training_inputs, training_results))

    # Перетворення набору даних тестування
    test_inputs = test_images.reshape(test_images.shape[0], 784) / 255.0
    test_results = keras.utils.to_categorical(test_labels, 10)
    test_data = list(zip(test_inputs, test_results))

    return training_data, test_data


# Создание модели нейронной сети
model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    training_data, test_data = load_data_wrapper()

    x_train, y_train = zip(*training_data)
    x_test, y_test = zip(*test_data)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Обучение модели
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    # Оценка точности
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
