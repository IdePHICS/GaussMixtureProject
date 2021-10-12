import numpy as np
from scipy.special import erfc

def gaussian(x, mean=0, var=1):
    return  np.exp(-1/(2*var) * (x-mean)**2)/np.sqrt(2*np.pi*var)
