import numpy as np
from numpy.random import randn

class RNN:
    def __init__(self, input_size, output_size, hidden_size=32):
        # Weights
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        y = self.Why @ h + self.by
        return y, h

    def backprop(self, d_y, learn_rate=0.07):
        n = len(self.last_inputs)

        # Calculate dL/dWhyd dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Initialize dL/dWhh, dL/dWxh,d dL/dbh to zero.
        d_Whh, d_Wxh, d_bh = np.zeros_like(self.Whh), np.zeros_like(self.Wxh), np.zeros_like(self.bh)

        # Calculate dL/dh for the last h.
        # dL/dh = dL/dy * dy/dh
        d_h = self.Why.T @ d_y

        # Initialize gradient for previous time step
        d_h_prev = np.zeros_like(d_h)

        # Backpropagate through time.
        for t in reversed(range(n)):
            temp = (1 - self.last_hs[t + 1] ** 2) * d_h
            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T
            d_h = self.Whh.T @ temp
            d_h += d_h_prev  # Add gradient from previous time step
            d_h_prev = d_h  # Cache gradient for the next iteration

        # Gradient clipping (you might prefer gradient scaling here)
        clip_value = 5.0
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -clip_value, clip_value, out=d)

        # Update weights and biases using gradient descent.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by
