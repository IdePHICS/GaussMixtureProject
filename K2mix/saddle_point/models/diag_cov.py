import numpy as np
from .base_model import Model
from ..losses.square_loss import SquareLoss
from ..losses.logistic_loss import LogisticLoss
from ..losses.hinge_loss import HingeLoss
from ..penalties.block_covariance import Diagonal, BlockDiagonal, DiagonalSparseMean

def initialise_loss(loss, sample_complexity, probability):
    if loss == 'square':
        return SquareLoss(sample_complexity = sample_complexity,
                          probability = probability)

    elif loss == 'logistic':
        return LogisticLoss(sample_complexity = sample_complexity,
                          probability = probability)

    elif loss == 'hinge':
        return HingeLoss(sample_complexity = sample_complexity,
                          probability = probability)
    else:
        raise NotImplementedError

class DiagModel(Model):
    '''
    Implements updates for a model with random means and diagonal covariances.
    '''
    def __init__(self, *, sample_complexity, regularisation, loss, penalty, probability, variance):

        self.channel = initialise_loss(loss, sample_complexity, probability)
        self.prior = Diagonal(penalty = penalty,
                              regularisation = regularisation,
                              variance = variance)

    def get_info(self):
        info = self.channel.get_info()
        info.update(self.prior.get_info())
        return info

    def update_se(self, V, q, m):
        Vhat, qhat, mhat, b = self.channel._update_hatoverlaps(V, q, m)
        Vnew, qnew, mnew = self.prior._update_overlaps(Vhat, qhat, mhat)
        return Vnew, qnew, mnew, b

class BlockModel(Model):
    '''
    Implements updates for a model with random means and diagonal covariances.
    '''
    def __init__(self, *, sample_complexity, regularisation, loss, penalty, probability, variances, ratios):

        self.channel = initialise_loss(loss, sample_complexity, probability)
        self.prior = BlockDiagonal(penalty = penalty,
                                   regularisation = regularisation,
                                   variances = variances,
                                   ratios = ratios)

    def get_info(self):
        info = self.channel.get_info()
        info.update(self.prior.get_info())
        return info

    def update_se(self, V, q, m):
        Vhat, qhat, mhat, b = self.channel._update_hatoverlaps(V, q, m)
        Vnew, qnew, mnew = self.prior._update_overlaps(Vhat, qhat, mhat)
        return Vnew, qnew, mnew, b

class CorrModel(Model):
    '''
    Implements updates for a model with
    - two clusters with same (diagonal) covariance
    - mean component equal to 0 in correspondence of element delta0 on the diagonal of the covariances
    - mean component equal to normal with zero mean and variance 1 in correspondence of element delt1 on the diagonal of the covariances
    '''
    def __init__(self, *, sample_complexity, regularisation, loss, penalty, probability, variances, ratios):
        if(penalty!='l1' and penalty!='l2'):
            raise NotImplementedError
        self.channel = initialise_loss(loss, sample_complexity, probability)
        self.prior = DiagonalSparseMean(regularisation = regularisation,
                                   penalty=penalty,
                                   variances = variances,
                                   ratios = ratios)

    def get_info(self):
        info = self.channel.get_info()
        info.update(self.prior.get_info())
        return info

    def update_se(self, V, q, m):
        Vhat, qhat, mhat, b = self.channel._update_hatoverlaps(V, q, m)
        Vnew, qnew, mnew = self.prior._update_overlaps(Vhat, qhat, mhat)
        return Vnew, qnew, mnew, b
