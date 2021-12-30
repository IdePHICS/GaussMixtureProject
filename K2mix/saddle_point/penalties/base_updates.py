from scipy.integrate import quad
from scipy.special import erfc
import numpy as np

def gaussian(x, mean=0, var=1):
    return np.exp(-.5*(x-mean)**2/var) / np.sqrt(2*np.pi*var)

class L2Base(object):
    '''
    Implements updates for l2 penalty with random means.
    '''
    def __init__(self, *, regularisation):
        self.lamb = regularisation

    def update_v(self, delta, Vhat, qhat, mhat):
        return delta/(self.lamb + Vhat * delta)

    def update_q(self, delta, Vhat, qhat, mhat):
        return delta * (delta * qhat + mhat**2)/(self.lamb + Vhat * delta)**2

    def update_m(self, delta, Vhat, qhat, mhat):
        return mhat/(self.lamb + Vhat * delta)

class L2Fixed(object):
    '''
    Implements updates for l2 penalty with random means.
    '''
    def __init__(self, *, regularisation,variance):
        self.lamb = regularisation
        self.var = variance

    def update_v(self, delta, Vhat, qhat, mhat):
        return delta/(self.lamb + Vhat * delta)

    def update_q(self, delta, Vhat, qhat, mhat):
        return delta * (delta * qhat + self.var*mhat**2)/(self.lamb + Vhat * delta)**2

    def update_m(self, delta, Vhat, qhat, mhat):
        return self.var*mhat/(self.lamb + Vhat * delta)
        
class L1Base(object):
    '''
    Implements updates for l1 penalty with random means.
    '''
    def __init__(self, *, regularisation):
        self.lamb = regularisation

    def update_v(self, delta, Vhat, qhat, mhat):
        def integrand(z):
            x = self.lamb + mhat*z
            return gaussian(z) * erfc(x/np.sqrt(2*qhat*delta)) / Vhat

        return quad(integrand, -10, 10)[0]

    def update_q(self, delta, Vhat, qhat, mhat):
        def integrand(z):
            x = self.lamb + mhat*z
            return gaussian(z) * (qhat*delta + x**2) * erfc(x/np.sqrt(2*qhat*delta))

        integral = quad(integrand, -10, 10)[0]
        other = - np.sqrt(2/(np.pi * (mhat**2+delta*qhat))) * (self.lamb*(qhat*delta)**2)/(mhat**2+delta*qhat) * np.exp(-.5 * self.lamb**2/(mhat**2+delta*qhat))

        return 1/delta * (integral + other)/Vhat**2

    def update_m(self, delta, Vhat, qhat, mhat):
        def integrand(z):
            x = self.lamb + mhat*z

            return gaussian(z) * x * z * erfc(x/np.sqrt(2*qhat*delta))

        integral = quad(integrand, -10, 10)[0]
        other =  np.sqrt(2/(np.pi * (mhat**2+delta*qhat))) * (self.lamb*mhat*qhat*delta)/(mhat**2+delta*qhat) * np.exp(-.5 * self.lamb**2/(mhat**2+delta*qhat))

        return (integral + other) / (Vhat*delta)

class L1Fixed(object):
    '''
    Implements updates for l1 penalty with fixed mean component (possibly zero).
    '''
    def __init__(self, *, regularisation,media):
        self.lamb = regularisation
        self.mu = media

    def update_v(self, delta, Vhat, qhat, mhat):
        def phi0(v,u,k):
            return 0.5*erfc((self.lamb+k*v)*1./np.sqrt(2.*u))
        return (phi0(mhat * self.mu,delta*qhat,1)+phi0(mhat * self.mu,delta*qhat,-1))*1./Vhat

    def update_q(self, delta, Vhat, qhat, mhat):
        def phi2(v,u,k):
            vm=self.lamb+k*v
            return -np.sqrt(u/(2.*np.pi))*np.exp(-vm**2/(2.*u))*vm+0.5*(u+vm**2)*erfc(vm/np.sqrt(2.*u))
        return (phi2(mhat*self.mu,qhat*delta,1)+phi2(mhat*self.mu,qhat*delta,-1))*1./(delta*Vhat**2)

    def update_m(self, delta, Vhat, qhat, mhat):
        def phi1(v,u,k):
            vm=self.lamb+k*v
            return np.sqrt(u/(2.*np.pi))*np.exp(-vm**2/(2.*u))+0.5*vm*erfc(vm/np.sqrt(2.*u))
        return self.mu*(phi1(mhat*self.mu,qhat*delta,-1)-phi2(mhat*self.mu,qhat*delta,1))*1./(delta*Vhat)
