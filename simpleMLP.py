import numpy as np
import math

class NaiveNN:
    def __init__(self, ws=None, bs=None):
        self._ws = ws
        self._bs = bs

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    sigmoid_v=np.vectorize(sigmoid)

    @staticmethod
    # hidden_dim is the hidden units m
    def fit(self, x, y, hidden_dim=20, lr=1e-3, epoch=1000):
        input_dim, output_dim = x.shape[1], y.shape[1]

        if self._ws is None:
            self._ws = [
                np.random.random([input_dim, hidden_dim]),
                np.random.random([hidden_dim, output_dim])]

        if self._bs is None:
            self._bs=[
                np.random.random([1,2])]
        losses = []
        for _ in range(epoch):
            # forward pass
            h = x.dot(self._ws[0])
            h_relu = NaiveNN.sigmoid_v(h)
            y_pred = h_relu.dot(self._ws[1])
            # np.linalg.norm(求范数)
            losses.append(np.linalg.norm(y_pred - y, ord="fro"))

            # backford pass
            # ∂L/∂y_pred
            d1 = 2 * (y_pred - y)
            # ∂L/∂w2 = ∂y_pred/∂w2* ∂L/∂y_pred
            # ∂y_pred/∂w2= h_relu.T
            dw2 = h_relu.T.dot(d1)
            # ∂L/∂w2 = ∂H/∂w2* ∂L/∂H
            # ∂L/∂H = ∂L/∂y_pred * w2^T * relu'
            dw1 = x.T.dot(d1.dot(self._ws[1].T) * (h_relu != 0))

            # uodate w
            self._ws[0] -= lr * dw1
            self._ws[1] -= lr * dw2

        return losses

    def predict(self, x):
        h = x.dot(self._ws[0])
        h_relu = NaiveNN.sigmoid_v(h)
        y_pred = h_relu.dot(self._ws[1])
        for result in y_pred:
            maxi=max(result)
            result[result<maxi]=0
            result[result==maxi]=1


        return y_pred
