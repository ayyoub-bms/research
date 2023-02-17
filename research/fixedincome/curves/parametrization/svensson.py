import numpy as np

from .parametrization import Parametrization


class Svensson(Parametrization):

    def __init__(self, beta_0=0, beta_1=0, beta_2=0,
                 beta_3=0, tau_1=0, tau_2=0):
        super().__init__()
        self.b0 = beta_0
        self.b1 = beta_1
        self.b2 = beta_2
        self.b3 = beta_3
        self.t1 = tau_1
        self.t2 = tau_2

    def instantaneous_forward_curve(self, theta):
        tt1 = theta / self.t1
        tt2 = theta / self.t2
        ifr = self.b0 + (self.b1 + self.b2 * tt1) * np.exp(-tt1)
        ifr += self.b3 * tt2 * np.exp(-tt2)
        return ifr

    def ifr_derivative(self, theta):
        tt1 = theta / self.t1
        tt2 = theta / self.t2
        ifrd = - self.b1 / self.t1 * np.exp(-tt1)
        ifrd += self.b2 * (1 - tt1) * np.exp(-tt1) / self.t1
        ifrd += self.b3 * (1 - tt2) * np.exp(-tt2) / self.t2
        return ifrd

    def spot_curve(self, theta):
        tt1 = theta / self.t1
        tt2 = theta / self.t2
        spot = self.b0 + self.b1 * (1 - np.exp(-tt1)) / tt1
        spot += self.b2 * ((1-np.exp(-tt1)) / tt1 - np.exp(-tt1))
        spot += self.b3 * ((1-np.exp(-tt2)) / tt2 - np.exp(-tt2))
        return spot

    def discount_curve(self, theta):
        theta = np.asarray([theta])
        if len(theta.shape) == 1:
            if theta[0] == 0:
                return 1
            return np.exp(-theta * self.spot_curve(theta)/100)
        else:
            theta = np.squeeze(theta)
        if theta[0] == 0:
            retval = np.ones(len(theta))
            retval[1:] = np.exp(-theta[1:] * self.spot_curve(theta[1:])/100)
            return retval
        else:
            return np.exp(-theta * self.spot_curve(theta)/100)

    def _update_params(self, args):
        self.b0 = args[0]
        self.b1 = args[1]
        self.b2 = args[2]
        self.b3 = args[3]
        self.t1 = args[4]
        self.t2 = args[5]

    def __repr__(self):
        return f"""
        Svensson curve parametrization:
        beta_0 = {self.b0}
        beta_1 = {self.b1}
        beta_2 = {self.b2}
        beta_3 = {self.b3}
        tau_1 = {self.t1}
        tau_2 = {self.t2}
        """
