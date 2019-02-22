import numpy as np
from numpy import log, sqrt, pi
from scipy.special import kv, gammaln


class LaplacianEigenpairs(object):
    name = 'Laplacian'

    def get_eigenpairs(self, x, basis_id, basis_interval=None, per_dimension=False):
        x_dim = x.shape[1]
        if basis_interval is None:
            basis_interval = np.max(np.abs(x), axis=0)
        if len(basis_interval) != x_dim:
            raise ValueError('Basis interval should have the same dimensionality as the input.')
        eigen_function, eigen_value = self._learn(x, basis_interval, basis_id)
        if per_dimension is True:
            return eigen_function, eigen_value
        else:
            return np.prod(eigen_function, axis=1), np.sum(eigen_value)

    @staticmethod
    def _learn(x, basis_interval, basis_id):
        n_smaps = x.shape[0]
        x_dim = x.shape[1]
        phi_x = np.ones((n_smaps, x_dim))
        lambda_ = np.zeros(x_dim)
        for k in range(x_dim):
            L_k = basis_interval[k]
            pi = np.pi
            j_k = basis_id
            x_k = x[:, k]
            normalization = 1./np.sqrt(L_k)
            up_ = pi * j_k * (x_k + L_k)
            down_ = 2 * L_k
            phi_x[:, k] = normalization*np.sin(up_/down_)
            lambda_[k] = np.power((pi * j_k)/(2*L_k), 2)
        return phi_x, lambda_


class MaternKernel(object):
    name = 'Matern'

    def __init__(self, nu=1, l=1, sf=1):
        self.nu = nu
        self.l = l
        self.sf = sf

    def log_kernel(self, r):
        return self._matern_kernel(r)

    def kernel(self, r):
        return np.exp(self._matern_kernel(r))

    def log_spectral(self, s):
        return self._matern_spectral(s)

    def spectral(self, s):
        return np.exp(self._matern_spectral(s))

    def estimate_kernel(self, phi_x1, phi_x2, lambdas):
        n_basis = len(lambdas)
        est_kernel = np.zeros(phi_x1.shape[0])
        for p in range(n_basis):
            est_kernel += self.spectral(sqrt(lambdas[p])) * phi_x1[:, p] * phi_x2[:, p]
        return est_kernel

    def _matern_kernel(self, r):
        nu = self.nu
        l = self.l
        sf = self.sf
        log_arg = log(sqrt(2*nu) * r) - log(l)
        arg = np.exp(log_arg)
        log_const = (1-nu)*log(2) - gammaln(nu)
        log_power_term = nu * log_arg
        #   TODO: what if arg is too large? in that case we need log_bessel not log of bessel
        log_bessel_term = log(kv(nu, arg))
        log_K_r = log(sf) + log_const + log_power_term + log_bessel_term
        return log_K_r

    def _matern_spectral(self, s):
        nu = self.nu
        l = self.l
        sf = self.sf
        log_arg = log(2*nu) - 2*log(l)
        arg = np.exp(log_arg)
        log_const = (0.5*log(2*pi)) + (nu*log_arg)
        log_gamma_term = gammaln(nu+0.5) - gammaln(nu)
        log_power_term = -(nu+.5) * log(arg + s**2)
        log_S_s = log(sf) + log_const + log_gamma_term + log_power_term
        return log_S_s

















