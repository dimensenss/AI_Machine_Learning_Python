import random  # бібліотека функцій для генерації випадкових значень


# Сторонні бібліотеки
import numpy as np  # бібліотека функцій для роботи з матрицями

def sigmoid(z): # визначення сигмоїдної функції активації
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):  # Похідна сигмоїдальної функції
    return sigmoid(z) * (1 - sigmoid(z))

# def sigmoid(x):
#     return x * (x > 0)
#
# def sigmoid_prime(x):
#     return 1. * (x > 0)

class Network(object):  # використовується для опису нейронної мережі

    def __init__(self, sizes):  # конструктор класу
        # задаємо кількість шарів
        self.sizes = sizes  # задаємо список розмірів шарів
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # задаємо
        self.weights = [np.random.randn(y, x) for x, y in
                        zip(sizes[:-1], sizes[1:])]  # задаємо випадкові початкові ваги зв'язків
        self.num_layers = len(self.sizes)



    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
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
            activation = sigmoid(z)  # зчитуємо вихідні сигнали поточного шару, застосовуючи сигмоїдальну функцію активації до активаційних потенціалів шару
            activations.append(activation)  # додаємо елемент (вихідні сигнали шару) в кінець списку

        # зворотне розповсюдження
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # зчитуємо міру впливу нейронів вихідного шару L на величину помилки (BP1)
        nabla_b[-1] = delta  # градієнт dC/db для шару L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градієнт dC / dw для шару L(BP4)

        for l in range(2, self.num_layers):
            z = zs[-l]  # активаційні потенціали l-го шару (рухаємось по  списку справа наліво)
            sp = sigmoid_prime(z)  # зчитуємо сигмоїдальну функцію від активаційних потенціалів l-го шару
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  #зчитуємо міру впливу нейронів l - го шару на величину помилки(BP2)
            nabla_b[-l] = delta  # градієнт dC/db для l-го шару (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #градієнт dC / dw для l - го шару(BP4)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):  # Оцінка прогресу в навчанні
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations,y):  # Обчислення приватних похідних функції вартості по вихідним сигналам останнього шару
        return (output_activations - y)




if __name__ == "__main__":
    from mnist_loader import load_data_wrapper
    training_data, validation_data, test_data = load_data_wrapper()


    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 30, 3, test_data=test_data)
