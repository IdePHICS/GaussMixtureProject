import numpy as np

from .base_updates import L2Base, L1Base, L1Fixed, L2Fixed
from .base_penalty import Penalty

def initialise_model(penalty, regularisation):
    if penalty == 'l2':
        return L2Base(regularisation = regularisation)
    elif penalty == 'l1':
        return L1Base(regularisation = regularisation)
    else:
        raise NotImplementedError

class Diagonal(Penalty):
    '''
    Implements updates for any penalty with Gaussian means and diagonal covariance.
    See base_model for details on modules.
    '''
    def __init__(self, *, penalty, regularisation, variance):
        self.lamb = regularisation
        self.delta = variance
        self.penalty = penalty

        self.updater = initialise_model(penalty, regularisation)

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'penalty': self.penalty,
            'covariance': 'diagonal',
            'problem': 'dense',
            'variance': self.delta,
            'regularisation': self.lamb
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        V = self.updater.update_v(self.delta, Vhat, qhat, mhat)
        q = self.updater.update_q(self.delta, Vhat, qhat, mhat)
        m = self.updater.update_m(self.delta, Vhat, qhat, mhat)

        return V, q, m

class BlockDiagonal(Penalty):
    '''
    Implements updates for any penalty with random means and block diagonal covariance.
    See base_model for details on modules.
    '''
    def __init__(self, *, penalty, regularisation, variances, ratios):
        self.lamb = regularisation
        self.deltas = variances
        self.ratios = ratios
        self.penalty = penalty

        self._check(variances, ratios)
        self.updater = initialise_model(penalty, regularisation)

    def _check(self, variances, ratios):
        '''
        Check if:
            - list of variances and ratios have the same size,
            - if ratios sum to one.
        '''
        if len(variances) != len(ratios):
            raise TypeError('List of must have same length as the ratios!')

        if np.sum(ratios) < 1-1e-5:
            raise TypeError('Ratios must sum to one.')


    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'penalty': self.penalty,
            'mean': 'gaussian',
            'problem': 'dense',
            'covariance': 'block',
            'n_blocks': len(self.deltas),
            'variances': list(self.deltas),
            'ratios': list(self.ratios),
            'regularisation': self.lamb
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        V, q, m = 0, 0, 0
        for k, delta in enumerate(self.deltas):
            V += self.ratios[k] * self.updater.update_v(delta, Vhat, qhat, mhat)
            q += self.ratios[k] * self.updater.update_q(delta, Vhat, qhat, mhat)
            m += self.ratios[k] * self.updater.update_m(delta, Vhat, qhat, mhat)

        return V, q, m


class DiagonalSparseMean(Penalty):
    '''
    Implements updates for the case of rho entries of the covariances corresponding to delta1 and gaussian mean element
    plus (1-rho) entries with delta0 corresponding to zero mean element
    '''
    def __init__(self, *, regularisation, penalty, variances, ratios):
        self.lamb = regularisation
        self.deltas = variances
        self.ratios = float(ratios)

        self._check()
        if(penalty=='l1'):
            self.updaterG = L1Base(regularisation=regularisation)
            self.updaterF = L1Fixed(regularisation=regularisation,media=0)
        elif(penalty=='l2'):
            self.updaterG = L2Base(regularisation=regularisation)
            self.updaterF = L2Fixed(regularisation=regularisation,variance=0)

    def _check(self):
        '''
        Check if the list of variances has 2 elements and that ratio is legit
        '''
        if len(self.deltas) != 2:
            raise TypeError('List of delta must have length 2!')

        if (self.ratios > 1 or self.ratios < 0 ) :
            raise TypeError('Ratios must between zero and one.')


    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'penalty': 'l1',
            'problem': 'sparse',
            'mean': 'Zero+Gaussian',
            'covariance': 'block',
            'n_blocks': len(self.deltas),
            'variances': list(self.deltas),
            'ratios': self.ratios,
            'regularisation': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        V = (self.ratios*self.updaterG.update_v(self.deltas[0], Vhat, qhat, mhat) +
                (1-self.ratios)*self.updaterF.update_v(self.deltas[1], Vhat, qhat, mhat))
        q = (self.ratios*self.updaterG.update_q(self.deltas[0], Vhat, qhat, mhat) +
                (1-self.ratios)*self.updaterF.update_q(self.deltas[1], Vhat, qhat, mhat))

        m = self.ratios * self.updaterG.update_m(self.deltas[0], Vhat, qhat, mhat)

        return V, q, m
