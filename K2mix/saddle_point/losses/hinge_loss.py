import numpy as np
from scipy.optimize import root, minimize_scalar
from scipy.special import erf, erfc

from ..auxiliary.errors import Q
from ..auxiliary.aux_functions import gaussian

from .base_loss import Loss

class HingeLoss(Loss):
    '''
    Implements updates for hinge loss.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, probability):
        self.alpha = sample_complexity
        self.prob = probability

    def get_info(self):
        info = {
            'loss': 'hinge',
            'sample_complexity': self.alpha,
            'probability': self.prob,
        }
        return info

    def _update_vhat(self, V, q, m, b):
        fun = lambda y: Q((1-V-m-y*b)/np.sqrt(q)) - Q((1-m-y*b)/np.sqrt(q))

        return 1/V * (self.prob * fun(1) + (1-self.prob) * fun(-1))

    def _update_qhat(self, V, q, m, b):
        def fun(y):
            mean = 1-m-y*b
            term1 = q*(mean * gaussian(0, mean=mean, var=q) -
                        (V+mean)*gaussian(V, mean=mean, var=q))

            term2 = (q + mean**2) * (Q(-mean/np.sqrt(q))-Q((V-mean)/np.sqrt(q)))
            term3 = V**2 * Q((V-mean)/np.sqrt(q))
            return term1+term2+term3

        return 1/V**2 * (self.prob * fun(1) + (1-self.prob) * fun(-1))

    def _update_mhat(self, V, q, m, b):
        def fun(y):
            mean = 1-m-y*b
            term1 = q * (gaussian(0, mean=mean, var=q) - gaussian(V, mean=mean, var=q))

            term2 = mean * (Q(-mean/np.sqrt(q)) - Q((V-mean)/np.sqrt(q)))
            term3 = V * Q((V-mean)/np.sqrt(q))

            return term1+term2+term3

        return 1/V * (self.prob * fun(1) + (1-self.prob) * fun(-1))

    def _update_bias(self, V, q, m):
        fun = lambda b: self._update_mhat(V, q, m, b)
        return root(fun, 0).x


    def _update_hatoverlaps(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        b = self._update_bias(V, q, m)

        Vhat = self.alpha * self._update_vhat(V, q, m, b)
        qhat = self.alpha * self._update_qhat(V, q, m, b)
        mhat = self.alpha * self._update_mhat(V, q, m, b)

        return Vhat, qhat, mhat, b


    def get_train_loss(self, V, q, m, b):
        '''
        Given overlaps, returns the training loss
        '''
        A = (1-m-V-b)/np.sqrt(q)
        D = (V-1+m-b)/np.sqrt(q)

        term1 = self.prob*(np.exp(-A**2/2)/np.sqrt(2*np.pi)+.5*A*(erf(A/np.sqrt(2))+1))
        term2 = (1-self.prob)*(np.exp(-D**2/2)/np.sqrt(2*np.pi)-.5*D*(erfc(D/np.sqrt(2))))

        return term1 + term2
