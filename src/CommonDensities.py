
import numpy as np
from computeRealBinghamConstant import logPartition_saddle as bc
from numpy import linalg


class BinghamFixedPrincipalVectors(object):

    def logdC(self, kappa):
        [_, rho] = bc(kappa)
        return rho

    def logC(self, kappa):
        [logC, _] = bc(kappa)
        return logC

    def cov(self, kappa, V):
        rho = self.logdC(kappa)
        cov = np.dot(rho*V, V.T)
        return cov

    def loglikelihood_posterior(self, kappa):
        rho = self.logdC(kappa)
        logC = self.logC(kappa)
        logpdf = np.dot(kappa, rho) - logC
        return logpdf

    def loglikelihood_prior(self, kappa, V, B):
        rho = self.logdC(kappa)
        dim = len(rho)
        logC = self.logC(kappa)
        logpdf = 0
        for d in range(dim):
            logpdf += rho[d] * np.trace(np.dot(B, np.dot(V[d, :], V[d, :])))
        logpdf = logpdf - logC
        return logpdf


class BinghamWatson(object):

    def logdC(self, kappa):
        [_, rho] = bc(kappa)
        return rho

    def logC(self, kappa):
        [logC, _] = bc(kappa)
        return logC

    def cov(self, kappa, V):
        rho = self.logdC(kappa)
        cov = np.dot(rho*V, V.T)
        return cov

    def loglikelihood_posterior(self, kappa, j):
        rho = self.logdC(kappa)
        logC = self.logC(kappa)
        logpdf = (kappa[j]*rho[j]) - logC
        return logpdf

    def loglikelihood_prior(self, kappa, V, B):
        rho = self.logdC(kappa)
        dim = len(rho)
        logC = self.logC(kappa)
        logpdf = 0
        for d in range(dim):
            logpdf += rho[d] * np.trace(np.dot(B, np.dot(V[d, :], V[d, :])))
        logpdf = logpdf - logC
        return logpdf


class Bingham(object):
    def __init__(self, B):
        eigen_values, eigen_vectors = linalg.eig(B)
        idx = eigen_values.argsort()[::-1]
        self.kappa = eigen_values[idx]
        self.axes = eigen_vectors[:, idx]
        [self.log_const, self.rho] = bc(self.kappa)

    def kappa(self):
        return self.kappa

    def axes(self):
        return self.axes

    def log_const(self):
        return self.log_const

    def rho(self):
        return self.rho

    def cov(self):
        return np.dot(self.rho*self.axes, self.axes.T)







