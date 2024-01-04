import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

# Створення словника слів з навчальних даних
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d унікальних слів знайдено' % vocab_size)

# Створення словників для відповідності індексу та слову
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# Функція для створення вектора входів з тексту
def createInputs(text):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

# Функція для визначення функції softmax
def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))

# Ініціалізація RNN з розміром словника та кількістю класів (у даному випадку 2)
rnn = RNN(vocab_size, 2)

# Функція для обробки навчальних даних та тестування
def processData(data, backprop=True):
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Обчислення градієнтів та навчання мережі
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)

# Навчання та тестування моделі протягом 1000 епох
for epoch in range(1000):
    # Навчання на навчальних даних
    train_loss, train_acc = processData(train_data)
    if epoch % 5 == 4:
      print('--- Епоха %d' % (epoch + 1))
      print('Навчання:\tВтрата %.4f | Точність: %.4f' % (train_loss, train_acc))

      # Тестування на валідаційних даних
      test_loss, test_acc = processData(test_data, backprop=False)
      print('Тест:\t\tВтрата %.4f | Точність: %.4f' % (test_loss, test_acc))
