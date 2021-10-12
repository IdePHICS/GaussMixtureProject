import numpy as np
from scipy.special import erfc

def Q(x):
    return (1/2) * erfc(x/np.sqrt(2))

def get_gen_error(q, m, b, prob):
    '''
    Given overlaps, returns the generalisation error
    '''
    gen_error = prob * Q((m + b)/np.sqrt(q)) + (1-prob) * Q((m-b)/np.sqrt(q))

    return gen_error
