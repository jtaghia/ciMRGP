import numpy as np
from scipy.special import psi, gammaln
from scipy.optimize import fsolve
from scipy.misc import logsumexp


class Stats(object):
    def __init__(self, posterior):
        qd = posterior

        self.noise_region_specific = qd.noise_region_specific
        self.bias_region_specific = qd.bias_region_specific

        self.latent_f_mean = None
        self.latent_f_var = None

        self.n_basis = qd.n_basis
        self.dy = qd.dy
        self.n_regions = qd.n_regions

    #   SCALE
        self.scale_axis_mean = []
        self.scale_moment2 = []
        self.scale_axis_central_moment2 = []
        for l in range(self.n_regions):
            self.scale_axis_mean.append(np.zeros((self.dy, self.n_basis)))
            self.scale_moment2.append(np.zeros(self.n_basis))
            self.scale_axis_central_moment2.append(np.zeros(self.n_basis))

    #   NOISE
        if qd.noise_region_specific is True:
            self.noise_mean = []
            self.noise_log_mean = []
            for l in range(self.n_regions):
                self.noise_mean.append(qd.noise_gamma_shape[l]/qd.noise_gamma_scale[l])
                self.noise_log_mean.append(psi(qd.noise_gamma_shape[l]) - np.log(qd.noise_gamma_scale[l]))
        elif qd.noise_region_specific is False:
            self.noise_mean = qd.noise_gamma_shape/qd.noise_gamma_scale
            self.noise_log_mean = psi(qd.noise_gamma_shape) - np.log(qd.noise_gamma_scale)
        else:
            raise TypeError('noise_region_specific condition can be either True or False!')

    #   BIAS
        if qd.bias_region_specific is True:
            self.bias_mean = []
            self.bias_var = []
            for l in range(self.n_regions):
                self.bias_mean.append(qd.bias_normal_mean[l])
                self.bias_var.append(qd.bias_normal_precision[l]**-1)
        elif qd.bias_region_specific is False:
            self.bias_mean = qd.bias_normal_mean
            self.bias_var = qd.bias_normal_precision**-1
        else:
            raise TypeError('bias_region_specific condition can be either True or False!')

    #   LATENT FUNCTIONS
    def initialize_latent_functions(self, n_samps):
        self.latent_f_mean = []
        self.latent_f_var = []
        for l in range(self.n_regions):
            self.latent_f_mean.append(np.zeros((n_samps[l], self.dy)))
            self.latent_f_var.append(np.zeros(n_samps[l]))

    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:
    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:

    def update_scale(self, posterior, stats):
        self._update_scale_axis_mean(posterior, stats)
        self._update_scale_moment2(posterior, stats)
        self._update_scale_axis_central_moment2(posterior, stats)

    def _update_scale_axis_mean(self, posterior, stats):
        qd = posterior
        axis_cov = stats.axis_cov
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                self.scale_axis_mean[l][:, i] = qd.scale_mean_zeta[l][i] * \
                        np.dot(axis_cov[i, :, :], qd.scale_mean_y_tilde[l][:, i])

    def _update_scale_moment2(self, posterior, stats):
        qd = posterior
        axis_cov = stats.axis_cov
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                zeta2 = qd.scale_mean_zeta[l][i]**2
                y_tilde_il = qd.scale_mean_y_tilde[l][:, i]
                self.scale_moment2[l][i] = (1/qd.scale_precision[l][i]) + \
                    (zeta2 * np.trace(np.dot(np.outer(y_tilde_il, y_tilde_il), axis_cov[i, :, :])))

    def _update_scale_axis_central_moment2(self, posterior, stats):
        qd = posterior
        axis_cov = stats.axis_cov
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                term_ = axis_cov[i, :, :] - np.dot(axis_cov[i, :, :], axis_cov[i, :, :])
                zeta2 = qd.scale_mean_zeta[l][i]**2
                y_tilde_il = qd.scale_mean_y_tilde[l][:, i]
                yy_tilde = np.outer(y_tilde_il, y_tilde_il)
                self.scale_axis_central_moment2[l][i] = (1/qd.scale_precision[l][i]) + \
                                                        (zeta2 * np.trace(np.dot(yy_tilde, term_)))

    def update_noise(self, posterior):
        qd = posterior
        if qd.noise_region_specific is True:
            for l in range(self.n_regions):
                self.noise_mean[l] = qd.noise_gamma_shape[l]/qd.noise_gamma_scale[l]
                self.noise_log_mean[l] = psi(qd.noise_gamma_shape[l]) - np.log(qd.noise_gamma_scale[l])
        elif qd.noise_region_specific is False:
            self.noise_mean = qd.noise_gamma_shape/qd.noise_gamma_scale
            self.noise_log_mean = psi(qd.noise_gamma_shape) - np.log(qd.noise_gamma_scale)
        else:
            raise TypeError('Unsupported condition for noise_region_specific.')

    def update_bias(self, posterior):
        qd = posterior
        if qd.bias_region_specific is True:
            for l in range(self.n_regions):
                self.bias_mean[l] = qd.bias_normal_mean[l]
                self.bias_var[l] = qd.bias_normal_precision[l]**-1
        elif qd.bias_region_specific is False:
            self.bias_mean = qd.bias_normal_mean
            self.bias_var = qd.bias_normal_precision**-1
        else:
            raise TypeError('Unsupported condition for bias_region_specific.')

    def update_latent_functions(self, resolution, index_set, stats, phi_x):
        n_samps_0 = len(index_set[0][0])
        latent_f_mean = np.zeros((n_samps_0, self.dy))
        latent_f_var = np.zeros(n_samps_0)
        #   TODO: which one?
        for jp in range(resolution):
            stats_jp = stats[jp]
            phi_x_jp = phi_x[jp]
            latent_f_mean_temp = []
            latent_f_var_temp = []
            for l in range(stats_jp.n_regions):
                if stats_jp.bias_region_specific is True:
                    bias_mean = stats_jp.bias_mean[l]
                    bias_var = stats_jp.bias_var[l]
                else:
                    bias_mean = stats_jp.bias_mean
                    bias_var = stats_jp.bias_var
                n_samps = phi_x_jp[l].shape[0]
                sum_term_mean = np.zeros((n_samps, stats_jp.dy))
                sum_term_var = np.zeros(n_samps)
                for i in range(self.n_basis):
                    sum_term_mean += stats_jp.scale_axis_mean[l][:, i] * \
                                 np.tile(phi_x_jp[l][:, i], (self.dy, 1)).T
                    sum_term_var += (phi_x_jp[l][:, i]**2) * stats_jp.scale_axis_central_moment2[l][i]
                latent_f_mean_temp.append(bias_mean + sum_term_mean)
                latent_f_var_temp.append(bias_var + sum_term_var)
            latent_f_mean += np.concatenate(latent_f_mean_temp)
            latent_f_var += np.concatenate(latent_f_var_temp)
        latent_f_var = np.atleast_2d(latent_f_var).T
        for region in range(len(index_set[resolution])):
            self.latent_f_mean[region] = latent_f_mean[index_set[resolution][region], :]
            self.latent_f_var[region] = latent_f_var[index_set[resolution][region], :]

#   ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#   ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


class IndependentStats(object):
    def __init__(self, posterior):
        qd = posterior
        self.n_basis = qd.n_basis
        self.dy = qd.dy
        self.n_regions = qd.n_regions

        #   AXIS
        self.axis_cov = []
        for l in range(self.n_regions):
            self.axis_cov.append(np.zeros((self.n_basis, self.dy, self.dy)))

        #   ARD
        self.ard_mean = []
        self.ard_log_mean = []
        for l in range(self.n_regions):
            self.ard_mean.append(qd.ard_gamma_shape[l] / qd.ard_gamma_scale[l])
            self.ard_log_mean.append(psi(qd.ard_gamma_shape[l]) - np.log(qd.ard_gamma_scale[l]))

        #   PERMUTATION ALIGNMENT
        self.omega = []
        for l in range(self.n_regions):
            self.omega.append(np.ones((self.n_basis, self.n_basis)) / self.n_basis)

        self.noise_region_specific = qd.noise_region_specific
        self.bias_region_specific = qd.bias_region_specific

        self.latent_f_mean = None
        self.latent_f_var = None

    #   SCALE
        self.scale_axis_mean = []
        self.scale_moment2 = []
        self.scale_axis_central_moment2 = []
        for l in range(self.n_regions):
            self.scale_axis_mean.append(np.zeros((self.dy, self.n_basis)))
            self.scale_moment2.append(np.zeros(self.n_basis))
            self.scale_axis_central_moment2.append(np.zeros(self.n_basis))

    #   NOISE
        if qd.noise_region_specific is True:
            self.noise_mean = []
            self.noise_log_mean = []
            for l in range(self.n_regions):
                self.noise_mean.append(qd.noise_gamma_shape[l]/qd.noise_gamma_scale[l])
                self.noise_log_mean.append(psi(qd.noise_gamma_shape[l]) - np.log(qd.noise_gamma_scale[l]))
        elif qd.noise_region_specific is False:
            self.noise_mean = qd.noise_gamma_shape/qd.noise_gamma_scale
            self.noise_log_mean = psi(qd.noise_gamma_shape) - np.log(qd.noise_gamma_scale)
        else:
            raise TypeError('noise_region_specific condition can be either True or False!')

    #   BIAS
        if qd.bias_region_specific is True:
            self.bias_mean = []
            self.bias_var = []
            for l in range(self.n_regions):
                self.bias_mean.append(qd.bias_normal_mean[l])
                self.bias_var.append(qd.bias_normal_precision[l]**-1)
        elif qd.bias_region_specific is False:
            self.bias_mean = qd.bias_normal_mean
            self.bias_var = qd.bias_normal_precision**-1
        else:
            raise TypeError('bias_region_specific condition can be either True or False!')

    #   LATENT FUNCTIONS
    def initialize_latent_functions(self, n_samps):
        self.latent_f_mean = []
        self.latent_f_var = []
        for l in range(self.n_regions):
            self.latent_f_mean.append(np.zeros((n_samps[l], self.dy)))
            self.latent_f_var.append(np.zeros(n_samps[l]))

    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:
    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:

    #   UPDATE AXIS
    def update_axis(self, posterior):
        qd = posterior
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                sum_d = np.zeros((self.dy, self.dy))
                for d in range(self.dy):
                    sum_d += qd.axis_bingham_rho[l][i, d] * \
                             np.outer(qd.axis_bingham_axes[l][i, :, d], qd.axis_bingham_axes[l][i, :, d])
                self.axis_cov[l][i, :, :] = sum_d

    #   UPDATE ARD
    def update_ard(self, posterior):
        qd = posterior
        for l in range(self.n_regions):
            self.ard_mean[l] = qd.ard_gamma_shape[l]/qd.ard_gamma_scale[l]
            self.ard_log_mean[l] = psi(qd.ard_gamma_shape[l]) - np.log(qd.ard_gamma_scale[l])

    def update_scale(self, posterior, stats):
        self._update_scale_axis_mean(posterior, stats)
        self._update_scale_moment2(posterior, stats)
        self._update_scale_axis_central_moment2(posterior, stats)

    def _update_scale_axis_mean(self, posterior, stats):
        qd = posterior
        axis_cov = stats.axis_cov
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                self.scale_axis_mean[l][:, i] = qd.scale_mean_zeta[l][i] * \
                        np.dot(axis_cov[l][i, :, :], qd.scale_mean_y_tilde[l][:, i])

    def _update_scale_moment2(self, posterior, stats):
        qd = posterior
        axis_cov = stats.axis_cov
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                zeta2 = qd.scale_mean_zeta[l][i]**2
                y_tilde_il = qd.scale_mean_y_tilde[l][:, i]
                self.scale_moment2[l][i] = (1/qd.scale_precision[l][i]) + \
                    (zeta2 * np.trace(np.dot(np.outer(y_tilde_il, y_tilde_il), axis_cov[l][i, :, :])))

    def _update_scale_axis_central_moment2(self, posterior, stats):
        qd = posterior
        axis_cov = stats.axis_cov
        for l in range(self.n_regions):
            for i in range(self.n_basis):
                term_ = axis_cov[l][i, :, :] - np.dot(axis_cov[l][i, :, :], axis_cov[l][i, :, :])
                zeta2 = qd.scale_mean_zeta[l][i]**2
                y_tilde_il = qd.scale_mean_y_tilde[l][:, i]
                yy_tilde = np.outer(y_tilde_il, y_tilde_il)
                self.scale_axis_central_moment2[l][i] = (1/qd.scale_precision[l][i]) + \
                                                        (zeta2 * np.trace(np.dot(yy_tilde, term_)))

    def update_noise(self, posterior):
        qd = posterior
        if qd.noise_region_specific is True:
            for l in range(self.n_regions):
                self.noise_mean[l] = qd.noise_gamma_shape[l]/qd.noise_gamma_scale[l]
                self.noise_log_mean[l] = psi(qd.noise_gamma_shape[l]) - np.log(qd.noise_gamma_scale[l])
        elif qd.noise_region_specific is False:
            self.noise_mean = qd.noise_gamma_shape/qd.noise_gamma_scale
            self.noise_log_mean = psi(qd.noise_gamma_shape) - np.log(qd.noise_gamma_scale)
        else:
            raise TypeError('Unsupported condition for noise_region_specific.')

    def update_bias(self, posterior):
        qd = posterior
        if qd.bias_region_specific is True:
            for l in range(self.n_regions):
                self.bias_mean[l] = qd.bias_normal_mean[l]
                self.bias_var[l] = qd.bias_normal_precision[l]**-1
        elif qd.bias_region_specific is False:
            self.bias_mean = qd.bias_normal_mean
            self.bias_var = qd.bias_normal_precision**-1
        else:
            raise TypeError('Unsupported condition for bias_region_specific.')

    def update_latent_functions(self, resolution, index_set, stats, phi_x):
        # if resolution > 0:
        n_samps_0 = len(index_set[0][0])
        latent_f_mean = np.zeros((n_samps_0, self.dy))
        latent_f_var = np.zeros(n_samps_0)
        #   TODO: which one?
        for jp in range(resolution):
            stats_jp = stats[jp]
            phi_x_jp = phi_x[jp]
            latent_f_mean_temp = []
            latent_f_var_temp = []
            for l in range(stats_jp.n_regions):
                if stats_jp.bias_region_specific is True:
                    bias_mean = stats_jp.bias_mean[l]
                    bias_var = stats_jp.bias_var[l]
                else:
                    bias_mean = stats_jp.bias_mean
                    bias_var = stats_jp.bias_var
                n_samps = phi_x_jp[l].shape[0]
                sum_term_mean = np.zeros((n_samps, stats_jp.dy))
                sum_term_var = np.zeros(n_samps)
                for i in range(self.n_basis):
                    sum_term_mean += stats_jp.scale_axis_mean[l][:, i] * \
                                 np.tile(phi_x_jp[l][:, i], (self.dy, 1)).T
                    sum_term_var += (phi_x_jp[l][:, i]**2) * stats_jp.scale_axis_central_moment2[l][i]
                latent_f_mean_temp.append(bias_mean + sum_term_mean)
                latent_f_var_temp.append(bias_var + sum_term_var)
            latent_f_mean += np.concatenate(latent_f_mean_temp)
            latent_f_var += np.concatenate(latent_f_var_temp)
        latent_f_var = np.atleast_2d(latent_f_var).T
        for region in range(len(index_set[resolution])):
            self.latent_f_mean[region] = latent_f_mean[index_set[resolution][region], :]
            self.latent_f_var[region] = latent_f_var[index_set[resolution][region], :]


#   ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#   ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

class SharedStats(object):
    def __init__(self, posterior):
        qd = posterior
        self.n_basis = qd.n_basis
        self.dy = qd.dy

    #   AXIS
        self.axis_cov = np.zeros((self.n_basis, self.dy, self.dy))
        self.axis_cov = np.zeros((self.n_basis, self.dy, self.dy))

    #   ARD
        self.ard_mean = qd.ard_gamma_shape/qd.ard_gamma_scale
        self.ard_log_mean = psi(qd.ard_gamma_shape) - np.log(qd.ard_gamma_scale)

    #   PERMUTATION ALIGNMENT
        self.omega = np.ones((self.n_basis, self.n_basis))/self.n_basis

    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:
    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:

    #   UPDATE AXIS
    def update_axis(self, posterior):
        qd = posterior
        for i in range(self.n_basis):
            sum_d = np.zeros((self.dy, self.dy))
            for d in range(self.dy):
                sum_d += qd.axis_bingham_rho[i, d] * \
                         np.outer(qd.axis_bingham_axes[i, :, d], qd.axis_bingham_axes[i, :, d])
            self.axis_cov[i, :, :] = sum_d

    #   UPDATE ARD
    def update_ard(self, posterior):
        qd = posterior
        self.ard_mean = qd.ard_gamma_shape/qd.ard_gamma_scale
        self.ard_log_mean = psi(qd.ard_gamma_shape) - np.log(qd.ard_gamma_scale)

    def update_omega(self, prior, stats):
        b_prime = prior.axis_bingham_b
        log_const_prime = prior.axis_bingham_log_const
        axis_cov = stats.axis_cov
        ard_shape_prime = prior.ard_gamma_shape
        ard_scale_prime = prior.ard_gamma_scale
        ard_log_mean = stats.ard_log_mean
        ard_mean = stats.ard_mean
        self.omega = self._get_omega(b_prime, log_const_prime, ard_shape_prime, ard_scale_prime,
                                     axis_cov, ard_log_mean, ard_mean, self.n_basis)

    #   UPDATE PERMUTATION
    @staticmethod
    def _get_omega(B_prime, log_const_prime, ard_shape_prime, ard_scale_prime, axis_cov, ard_log_mean,
                   ard_mean, n_basis):
        log_omega_hat = np.zeros((n_basis, n_basis))
        for i in range(n_basis):
            for k in range(n_basis):
                term1 = np.trace(np.dot(axis_cov[i, :, :], B_prime[k, :, :]))
                log_omega_hat[i, k] = term1 - log_const_prime[k] \
                    + ard_shape_prime[k]*np.log(ard_scale_prime[k]) - gammaln(ard_shape_prime[k]) \
                    + (ard_shape_prime[k] - 1)*ard_log_mean[i] \
                    - ard_scale_prime[k] * ard_mean[i]
        ln_eta_hat = fsolve(_func_omega, np.zeros(2*n_basis), log_omega_hat)
        omega = np.zeros((n_basis, n_basis))
        for i in range(n_basis):
            ln_alpha_hat = ln_eta_hat[0:n_basis]
            ln_beta_hat = ln_eta_hat[n_basis: 2*n_basis]
            for k in range(n_basis):
                omega[i, k] = np.exp(ln_alpha_hat[i] + ln_beta_hat[k] + log_omega_hat[i, k])
        return omega


def _func_omega(ln_eta, ln_omega):
    n_basis = ln_omega.shape[0]
    ln_alpha = ln_eta[0:n_basis]
    ln_beta = ln_eta[n_basis: 2*n_basis]
    ln_a = []
    for k in range(n_basis):
        ln_a_k = np.zeros(n_basis)
        for i in range(n_basis):
            ln_a_k[i] = ln_alpha[i] + ln_omega[i, k]
        ln_a.append(ln_a_k)

    ln_b = []
    for i in range(n_basis):
        ln_b_i = np.zeros(n_basis)
        for k in range(n_basis):
            ln_b_i[k] = ln_beta[k] + ln_omega[i, k]
        ln_b.append(ln_b_i)

    out = []
    for l in range(n_basis):
        out.append(ln_alpha[l] + logsumexp(ln_b[l]))
        out.append(ln_beta[l] + logsumexp(ln_a[l]))
    return out

















