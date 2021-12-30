import numpy as np
from .base_loss import Loss


class SquareLoss(Loss):
    '''
    Implements updates for ridge regression task.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, probability):
        self.alpha = sample_complexity
        self.prob = probability

    def get_info(self):
        info = {
            'loss': 'square',
            'sample_complexity': self.alpha,
            'probability': self.prob,
        }
        return info

    def _update_hatoverlaps(self, V, q, m):
        b = (2*self.prob-1) * (1-m)

        Vhat = self.alpha / (1+V)
        qhat = self.alpha * (self.prob * (1-m-b)**2 + (1-self.prob) * (1-m+b)**2 + q)/(1+V)**2
        mhat = self.alpha * (1-m-(2*self.prob-1) * b) / (1+V)

        return Vhat, qhat, mhat, b

    def get_train_loss(self, V, q, m, b):
        '''
        Given overlaps, returns the training loss
        '''
        c3 = (1/2) * self.prob * (1 + q + m**2 + b**2 + 2*m*b - 2*m - 2*b)/((1+V)**2)
        c4 = (1/2) * (1-self.prob) * (1 + q + m**2 + b**2 - 2*m*b - 2*m + 2*b)/((1+V)**2)

        return c3+c4
