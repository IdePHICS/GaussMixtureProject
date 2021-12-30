class Penalty(object):
    '''
    Base class for a regularisation function.
    -- args --
    regularisation: regularisation strength
    '''
    def __init__(self, *, regularisation, variance):
        self.lamb = regularisation
        self.delta = variance

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'penalty': 'generic',
            'regularisation': self.lamb,
            'variance': self.delta
        }
        return info

    def _update_overlaps(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        raise NotImplementedError
