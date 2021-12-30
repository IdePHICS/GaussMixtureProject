class Model(object):
    '''
    Base class for a model, which is composed of a loss and a penalty.
    '''
    def __init__(self, *, sample_complexity, regularisation, loss, penalty):
        self.alpha = sample_complexity
        self.lamb = regularisation
        self.loss = loss
        self.penalty = penalty

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'model': 'generic',
            'loss': 'loss',
            'penalty': 'penalty',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def update_se(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        raise NotImplementedError

    def get_test_error(self, q, m):
        '''
        Method for computing the test error from overlaps.
        '''
        raise NotImplementedError

    def get_train_loss(self, V, q, m):
        '''
        Method for computing the training loss from overlaps.
        '''
        raise NotImplementedError
