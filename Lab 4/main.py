import pandas as pd
import numpy as np

# Функція активації - сигмоїда
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Функція, яка визначає очікуваний вихід для навчання
def func(x):
    return 1 if x == 1 else 0

# Клас нейронної мережі
class NeuralNetwork:
    INPUT_DIM = 3  # Кількість входів мережі
    learning_rate = 0.1  # Швидкість навчання
    weights = np.random.rand(INPUT_DIM)  # Випадкова ініціалізація ваг

    def __init__(self):
        self.expected_output = None

    # Метод навчання мережі
    def learn(self, filename):
        training_data = pd.read_excel(filename)
        self.expected_output = training_data['output']
        training_data = np.asarray(training_data.drop('output', axis=1))

        # Цикл навчання
        for epoch in range(1000):
            for i in range(len(training_data)):
                output_sum = np.sum(np.multiply(training_data[i, :], self.weights))
                output_value = sigmoid(output_sum)

                error = self.expected_output[i] - output_value

                grad = output_value * (1 - output_value)

                # Оновлення ваг відповідно до правила навчання
                for n in range(self.INPUT_DIM):
                    self.weights[n] = self.weights[n] + self.learning_rate * error * training_data[i, n] * grad
        print("Weights:", self.weights)

    # Метод для тестування мережі на тестовому наборі даних
    def testing(self, filename):

        test_data = np.asarray(pd.read_excel(filename))
        count_errors = 0
        lst_errors = []

        for i in range(len(test_data)):
            output_sum = np.sum(np.multiply(test_data[i, :], self.weights))
            output_value = sigmoid(output_sum)

            real = func(test_data[i][1])

            print("Input: {}, Output: {}, Real: {}".format(test_data[i], output_value, real))
            if round(output_value) == real:
                continue
            else:
                lst_errors.append((test_data[i], output_value, real))
                count_errors += 1

        print('Помилка', count_errors / len(test_data))
        print("Ваги:", self.weights)
        if len(lst_errors) :print(lst_errors, sep='\n')

if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.learn('3D_data.xlsx')
    nn.testing('3D_data_test.xlsx')