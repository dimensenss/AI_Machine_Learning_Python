import numpy as np
from numpy.random import randn

class RNN:
    def __init__(self, input_size, output_size, hidden_size=32):
        # Ваги
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Зсуви (байаси)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = { 0: h }

        # Прохід вперед по часу
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=0.07):
        n = len(self.last_inputs)

        # Обчислення градієнтів dL/dWhy та dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Ініціалізація градієнтів dL/dWhh, dL/dWxh, dL/dbh нулями.
        d_Whh, d_Wxh, d_bh = np.zeros_like(self.Whh), np.zeros_like(self.Wxh), np.zeros_like(self.bh)

        # Обчислення градієнтів dL/dh для останнього h.
        # dL/dh = dL/dy * dy/dh
        d_h = self.Why.T @ d_y

        # Зворотнє поширення (backpropagation) у часі.
        for t in reversed(range(n)):
            # Проміжне значення: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # Обчислення градієнтів для Whh, Wxh
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T

            # Оновлення dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh.T @ temp

        # Обмеження градієнтів, щоб уникнути їхнього вибуху.
        clip_value = 0.5
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -clip_value, clip_value, out=d)

        # Оновлення ваг та зсувів за допомогою градієнтного спуску.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by
