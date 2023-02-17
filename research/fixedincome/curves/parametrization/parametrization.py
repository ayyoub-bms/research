import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize


class Parametrization:

    def instantaneous_forward_curve(self, theta):
        raise NotImplementedError()

    def discount_curve(self, theta):
        raise NotImplementedError()

    def ifr_derivative(self, theta):
        raise NotImplementedError()

    def spot_curve(self, theta):
        raise NotImplementedError()

    def _update_params(self, args):
        raise NotImplementedError()

    def plot_forward(self, **kwargs):
        theta = np.linspace(0, 30, 1000)
        plt.plot(theta, self.instantaneous_forward_curve(theta), **kwargs)
        plt.xlabel('Term to maturity')
        plt.ylabel('Instantaneous forward rates')
        plt.title('Svensson parametrization of instantaneous forward rates')

    def plot_spot(self, **kwargs):
        theta = np.linspace(0.00274, 30, 1000)
        plt.plot(theta, self.spot_curve(theta), **kwargs)
        plt.xlabel('Term to maturity')
        plt.ylabel('Spot rates')
        plt.title('Svensson parametrization of spot rates')

    def plot_discount(self, **kwargs):
        theta = np.linspace(0.00274, 30, 1000)
        plt.plot(theta, self.discount_curve(theta), **kwargs)
        plt.xlabel('Term to maturity')
        plt.ylabel('discount rates')
        plt.title('Svensson parametrization of the discount curve')

    def _objective(self, params, times, rates):
        self._update_params(params)
        estimates = self.instantaneous_forward_curve(times)
        return np.power(rates - estimates, 2).sum()

    def calibrate(self, times, rates, nb_trials=100):
        value = np.inf
        optimal_params = None
        options = {'ftol': 1e-8, 'maxiter': 5000}
        for i in range(1, nb_trials+1):
            print(f'\rCalibrating: iter = {i}', end='')
            params = 3 * np.random.rand(6)
            res = minimize(self._objective,
                           params,
                           method='L-BFGS-B',
                           options=options,
                           args=(times, rates))

            if res.fun < value:
                value = res.fun
                optimal_params = res.x
            if value < 1e-8:
                print(f'\nOptimal parameters found after {i} iterations')
                break
        self._update_params(optimal_params)
