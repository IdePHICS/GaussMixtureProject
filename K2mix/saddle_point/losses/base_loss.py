class Loss(object):
    '''
    Base class for a loss function.
    -- args --
    sample_complexity: sample complexity
    '''
    def __init__(self, *, sample_complexity, probability):
        self.alpha = sample_complexity
        self.prob = probability

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'loss': 'loss',
            'sample_complexity': self.alpha,
            'probability': self.prob,
        }
        return info

    def _update_hatoverlaps(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        raise NotImplementedError
