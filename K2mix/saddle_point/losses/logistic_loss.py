import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, minimize_scalar

from .base_loss import Loss
from ..auxiliary.aux_functions import gaussian

class LogisticLoss(Loss):
    '''
    Implements updates for logistic loss.
    See base_model for details on modules.
    '''
    def __init__(self, epsrel=1e-11, limit=100, int_boundary=10, *, sample_complexity, probability):
        self.alpha = sample_complexity
        self.prob = probability


        self.epsrel = epsrel
        self.limit = limit
        self.bdry = int_boundary

    def get_info(self):
        info = {
            'loss': 'logistic',
            'sample_complexity': self.alpha,
            'probability': self.prob,
        }
        return info

    def loss(self, z):
        '''
        Logistic loss
        '''
        return np.log(1+np.exp(-z))

    def d1loss(self, z):
        '''
        First derivative of logistic loss
        '''
        return -1/(1+np.exp(z))

    def d2loss(self, z):
        '''
        Second derivative of logistic loss
        '''
        return 1/(4*np.cosh(z/2)**2)

    def get_proximal(self, V, y, omega, b):
        # fun = lambda x: x - omega + V * y * self.d1loss(y*(x+b))
        # return root(fun, omega).x
        fun = lambda z: .5 * (z-omega)**2/V + self.loss(y*(z+b))
        return minimize_scalar(fun).x

    def _update_vhat(self, V, q, m, b):
        '''
        Updates the vhat parameter
        '''
        def integrand(z):
            omega1=np.sqrt(q)*z+m
            omega2=np.sqrt(q)*z-m

            prox1 = self.get_proximal(V, 1, omega1, b)
            prox2 = self.get_proximal(V, -1, omega2, b)

            return (self.prob / (1+V*self.d2loss(prox1+b)) + (1-self.prob)/(1+V*self.d2loss(-prox2-b))) * gaussian(z)

        return (1 - quad(integrand, -self.bdry, self.bdry, limit=self.limit, epsrel=self.epsrel)[0])/V

    def _update_qhat(self, V, q, m, b):
        '''
        Updates the vhat parameter
        '''
        def integrand(z):
            omega1=np.sqrt(q)*z+m
            omega2=np.sqrt(q)*z-m

            prox1 = self.get_proximal(V, 1, omega1, b)
            prox2 = self.get_proximal(V, -1, omega2, b)

            return ((self.prob*(self.d1loss(prox1+b)**2)) + (1-self.prob)*(self.d1loss(-prox2-b)**2)) * gaussian(z)

        return quad(integrand, -self.bdry, self.bdry, limit=self.limit, epsrel=self.epsrel)[0]

    def _update_mhat(self, V,q,m, b):
        '''
        Updates the vhat parameter
        '''
        def integrand(z):
            omega1=np.sqrt(q)*z+m
            omega2=np.sqrt(q)*z-m

            prox1 = self.get_proximal(V, 1, omega1, b)
            prox2 = self.get_proximal(V, -1, omega2, b)

            return  (self.prob * self.d1loss(prox1+b) + (1-self.prob) * self.d1loss(-prox2-b)) * gaussian(z)

        return -quad(integrand, -self.bdry, self.bdry, limit=self.limit, epsrel=self.epsrel)[0]

    def _update_bias(self, V, q, m):
        def bias_function(b):
            def integrand(z):
                omega1=np.sqrt(q)*z+m
                omega2=np.sqrt(q)*z-m

                prox1 = self.get_proximal(V, 1, omega1, b)
                prox2 = self.get_proximal(V, -1, omega2, b)

                return (self.prob * self.d1loss(prox1+b)+(1-self.prob) * self.d1loss(-prox2-b))*gaussian(z)

            return quad(integrand, -self.bdry, self.bdry, limit=self.limit, epsrel=self.epsrel)[0]

        return root(bias_function, 0).x


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
        def integrand(z):
            omega1=np.sqrt(q)*z+m
            omega2=np.sqrt(q)*z-m

            prox1 = self.get_proximal(V, 1, omega1, b)
            prox2 = self.get_proximal(V, -1, omega2, b)

            return (self.prob * self.loss(prox1+b)+(1-self.prob) * self.loss(-prox2-b)) * gaussian(z)

        return quad(integrand, -self.bdry, self.bdry, limit=self.limit, epsrel=self.epsrel)[0]
