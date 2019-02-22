import numpy as np
from SanityCheck import SanityCheck
from CommonDensities import Bingham
from numpy.linalg import norm

EPSILON = 1e-200


class Posterior(object):
    def __init__(self, prior):
        self.scale_mean_zeta = []
        self.scale_mean_y_tilde = []
        for l in range(prior.n_regions):
            self.scale_mean_zeta.append(np.zeros(prior.n_basis))
            self.scale_mean_y_tilde.append(np.zeros((prior.dy, prior.n_basis)))

        self.scale_precision = np.copy(prior.scale_precision)

        self.noise_region_specific = prior.noise_region_specific
        self.noise_gamma_scale = np.copy(prior.noise_gamma_scale)
        self.noise_gamma_shape = np.copy(prior.noise_gamma_shape)

        self.bias_region_specific = prior.bias_region_specific
        self.bias_normal_mean = np.copy(prior.bias_normal_mean)
        self.bias_normal_precision = np.copy(prior.bias_normal_precision)

        self.dy = prior.dy
        self.n_basis = prior.n_basis
        self.n_regions = prior.n_regions

    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:
    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:

    #   SCALE GIVEN AXIS
    def update_scale_given_axis(self, y_mean, phi_x, prior, stats, shared_stats, spectral_density):
        pd = prior
        n_regions = pd.n_regions
        for l in range(n_regions):
            if pd.noise_region_specific is True:
                self.scale_precision[l] = shared_stats.ard_mean/spectral_density[l] + \
                                          stats.noise_mean[l] * np.sum(phi_x[l]*phi_x[l], axis=0)
                self.scale_mean_zeta[l] = np.divide(stats.noise_mean[l], self.scale_precision[l])
            elif pd.noise_region_specific is False:
                self.scale_precision[l] = shared_stats.ard_mean/spectral_density[l] + \
                                          stats.noise_mean * np.sum(phi_x[l]*phi_x[l], axis=0)
                self.scale_mean_zeta[l] = np.divide(stats.noise_mean, self.scale_precision[l])
            else:
                raise TypeError('unsupported condition for noise_region_specific.')
            if pd.bias_region_specific is True:
                bias_mean = stats.bias_mean[l]
            elif pd.bias_region_specific is False:
                bias_mean = stats.bias_mean
            else:
                raise TypeError('unsupported condition for bias_region_specific.')
            self.scale_mean_y_tilde[l] = self._get_y_tilde(y_mean=y_mean[l],
                                                           phi_x=phi_x[l],
                                                           scale_axis_mean=stats.scale_axis_mean[l],
                                                           bias_mean=bias_mean,
                                                           f_bar=stats.latent_f_mean[l])

    def _get_y_tilde(self, y_mean, phi_x, scale_axis_mean, bias_mean, f_bar):
        y = y_mean
        y_tilde = np.zeros((self.dy, self.n_basis))
        for i in range(self.n_basis):
            sub_range = list(range(self.n_basis))
            del(sub_range[i])
            penalty_term = self._get_penalty_mean(sub_range, scale_axis_mean, phi_x)
            y_tilde_bar = f_bar + bias_mean + penalty_term
            y_tilde[:, i] = np.sum(np.tile(phi_x[:, i], (self.dy, 1)).T *
                                   (y - y_tilde_bar), axis=0)
        return y_tilde

    def _get_penalty_mean(self, subrange, scale_axis_mean, phi_x):
        n_samps = phi_x.shape[0]
        penalty = np.zeros((n_samps, self.dy))
        for k in subrange:
            penalty += scale_axis_mean[:, k] * np.tile(phi_x[:, k], (self.dy, 1)).T
        return penalty

    #   BIAS
    def update_bias_given_noise(self, y_mean, phi_x, prior, stats):
        if self.bias_region_specific is True:
            for l in range(self.n_regions):
                phi_x_l = phi_x[l]
                y_l = y_mean[l]
                n_samps = phi_x_l.shape[0]
                self.bias_normal_precision[l] = prior.bias_normal_precision[l] + n_samps
                sum_term = np.zeros_like(y_l)
                for i in range(self.n_basis):
                    sum_term += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
                self.bias_normal_mean[l] = np.power(self.bias_normal_precision[l], -1) * \
                    (prior.bias_normal_mean[l]*prior.bias_normal_precision[l] +
                     np.sum(y_l - sum_term - stats.latent_f_mean[l], axis=0))
        elif prior.bias_region_specific is False:
            n_samps_all_regions = 0
            for l in range(self.n_regions):
                n_samps_all_regions += phi_x[l].shape[0]
            self.bias_normal_precision = prior.bias_normal_precision + n_samps_all_regions
            sum_term = np.zeros(self.dy)
            for l in range(self.n_regions):
                phi_x_l = phi_x[l]
                y_l = y_mean[l]
                sum_term_l = np.zeros_like(y_l)
                for i in range(self.n_basis):
                    sum_term_l += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
                sum_term += np.sum(y_l - sum_term_l - stats.latent_f_mean[l], axis=0)
            self.bias_normal_mean = np.power(self.bias_normal_precision, -1) * \
                (prior.bias_normal_mean*prior.bias_normal_precision + sum_term)
        else:
            raise TypeError('Unknown condition for bias_region_specific.')

    #   NOISE
    def update_noise(self, y_mean, y_var, phi_x, prior, posterior, stats):
        if (self.noise_region_specific is True) and (self.bias_region_specific is True):
            self._update_regional_noise_regional_bias(y_mean, y_var,
                                                      phi_x, prior, posterior, stats)

        elif (self.noise_region_specific is True) and (self.bias_region_specific is False):
            self._update_regional_noise_shared_bias(y_mean, y_var,
                                                    phi_x, prior, posterior, stats)

        elif (self.noise_region_specific is False) and (self.bias_region_specific is True):
            self._update_shared_noise_regional_bias(y_mean, y_var,
                                                    phi_x, prior, posterior, stats)

        elif (self.noise_region_specific is False) and (self.bias_region_specific is False):
            self._update_shared_noise_shared_bias(y_mean, y_var,
                                                  phi_x, prior, posterior, stats)
        else:
            raise TypeError('Unknown conditions.')

    def _update_regional_noise_regional_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            self.noise_gamma_shape[l] = prior.noise_gamma_shape[l] + 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_l = y_var[l] #* n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            term3 = prior.bias_normal_precision[l] * np.power(norm(prior.bias_normal_mean[l]), 2)
            term4 = posterior.bias_normal_precision[l]*np.power(norm(posterior.bias_normal_mean[l]), 2)
            self.noise_gamma_scale[l] = prior.noise_gamma_scale[l] +\
                0.5*(term3 - term4 + mean_term_l + var_f_l + var_au_l + y_var_l)

    def _update_regional_noise_shared_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        term3 = prior.bias_normal_precision * np.power(norm(prior.bias_normal_mean), 2)
        term4 = posterior.bias_normal_precision*np.power(norm(posterior.bias_normal_mean), 2)
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            self.noise_gamma_shape[l] = prior.noise_gamma_shape[l] + 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_l = y_var[l] * n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            self.noise_gamma_scale[l] = prior.noise_gamma_scale[l] + \
                0.5*(term3 - term4 + mean_term_l + var_f_l + var_au_l + y_var_l)

    def _update_shared_noise_regional_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        noise_gamma_shape_update = 0
        noise_gamma_scale_update = 0
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            noise_gamma_shape_update += 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_l = y_var[l] * n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            term3 = prior.bias_normal_precision[l] * np.power(norm(prior.bias_normal_mean[l]), 2)
            term4 = posterior.bias_normal_precision[l]*np.power(norm(posterior.bias_normal_mean[l]), 2)
            noise_gamma_scale_update += 0.5*(term3 - term4 + mean_term_l + var_f_l + var_au_l + y_var_l)
        self.noise_gamma_shape = prior.noise_gamma_shape + noise_gamma_shape_update
        self.noise_gamma_scale = prior.noise_gamma_scale + noise_gamma_scale_update

    def _update_shared_noise_shared_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        term3 = prior.bias_normal_precision * np.power(norm(prior.bias_normal_mean), 2)
        term4 = posterior.bias_normal_precision*np.power(norm(posterior.bias_normal_mean), 2)
        mean_term = 0
        var_f = 0
        var_au = 0
        y_var_sum = 0
        noise_gamma_shape_update = 0
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            noise_gamma_shape_update += 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_sum += y_var[l]*n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term += np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f += np.sum(stats.latent_f_var[l])
            var_au += np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
        noise_gamma_scale_update = 0.5*(term3 - term4 + mean_term + var_f + var_au + y_var_sum)
        self.noise_gamma_shape = prior.noise_gamma_shape + noise_gamma_shape_update
        self.noise_gamma_scale = prior.noise_gamma_scale + noise_gamma_scale_update

#   ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#   ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


class IndependentPosterior(object):
    def __init__(self, prior):

        self.axis_bingham_b = np.copy(prior.axis_bingham_b)
        self.axis_bingham_kappa = np.copy(prior.axis_bingham_kappa)
        self.axis_bingham_rho = np.copy(prior.axis_bingham_rho)
        self.axis_bingham_axes = np.copy(prior.axis_bingham_axes)
        self.axis_bingham_log_const = np.copy(prior.axis_bingham_log_const)

        self.ard_gamma_shape = np.copy(prior.ard_gamma_shape)
        self.ard_gamma_scale = np.copy(prior.ard_gamma_scale)

        self.scale_mean_zeta = []
        self.scale_mean_y_tilde = []
        for l in range(prior.n_regions):
            self.scale_mean_zeta.append(np.zeros(prior.n_basis))
            self.scale_mean_y_tilde.append(np.zeros((prior.dy, prior.n_basis)))

        self.scale_precision = np.copy(prior.scale_precision)

        self.noise_region_specific = prior.noise_region_specific
        self.noise_gamma_scale = np.copy(prior.noise_gamma_scale)
        self.noise_gamma_shape = np.copy(prior.noise_gamma_shape)

        self.bias_region_specific = prior.bias_region_specific
        self.bias_normal_mean = np.copy(prior.bias_normal_mean)
        self.bias_normal_precision = np.copy(prior.bias_normal_precision)

        self.dy = prior.dy
        self.n_basis = prior.n_basis
        self.n_regions = prior.n_regions

    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:
    #   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:

    #   BASIS AXIS
    def update_axis(self, prior, posterior, stats):
        n_regions = posterior.n_regions
        n_basis = self.n_basis
        dy = self.dy
        for l in range(n_regions):
            for i in range(n_basis):
                sum_b = np.zeros((dy, dy))
                for k in range(n_basis):
                    sum_b += stats.omega[l][i, k]*prior.axis_bingham_b[l][k, :, :]
                sum_term = np.zeros((dy, dy))
                zeta_l = posterior.scale_mean_zeta[l][i]
                y_tilde_il = posterior.scale_mean_y_tilde[l][:, i]
                yy_tilde_zeta_l = zeta_l*np.outer(y_tilde_il, y_tilde_il)
                if posterior.noise_region_specific is True:
                    noise_mean = stats.noise_mean[l]
                elif posterior.noise_region_specific is False:
                    noise_mean = stats.noise_mean
                else:
                    raise TypeError('unsupported condition for noise_region_specific.')
                sum_term += (0.5 * noise_mean * yy_tilde_zeta_l)
                b = sum_b + sum_term
                sanity_check = SanityCheck()
                if sanity_check.isPD(b) is True:
                    self.axis_bingham_b[l][i, :, :] = b
                else:
                    self.axis_bingham_b[l][i, :, :] = sanity_check.nearestPD(b)
                bingham_pd = Bingham(self.axis_bingham_b[l][i, :, :])
                bingham_pd.kappa = np.real(bingham_pd.kappa)
                bingham_pd.kappa[bingham_pd.kappa < 0] = 0.
                self.axis_bingham_kappa[l][i, :] = bingham_pd.kappa
                self.axis_bingham_axes[l][i, :, :] = np.real(bingham_pd.axes)
                self.axis_bingham_rho[l][i, :] = np.real(bingham_pd.rho)
                self.axis_bingham_log_const[l][i] = np.real(bingham_pd.log_const)

    #   ARD
    def update_ard(self, prior, stats, spectral_density):
        for l in range(stats.n_regions):
            for i in range(self.n_basis):
                shape_prime = prior.ard_gamma_shape[l]
                scale_prime = prior.ard_gamma_scale[l]
                self.ard_gamma_shape[l][i] = np.sum(stats.omega[l][i, :]*shape_prime) + 0.5*stats.n_regions
                beta_term2 = stats.scale_moment2[l][i]/spectral_density[l][i]
                self.ard_gamma_scale[l][i] = np.sum(stats.omega[l][i, :]*scale_prime) + 0.5*beta_term2

    #   SCALE GIVEN AXIS
    def update_scale_given_axis(self, y_mean, phi_x, prior, stats, spectral_density):
        pd = prior
        n_regions = pd.n_regions
        # TODO: check if the prior spectral density should be changed or not based on the updated interval??
        for l in range(n_regions):
            if pd.noise_region_specific is True:
                self.scale_precision[l] = stats.ard_mean[l]/spectral_density[l] + \
                                          stats.noise_mean[l] * np.sum(phi_x[l]*phi_x[l], axis=0)
                self.scale_mean_zeta[l] = np.divide(stats.noise_mean[l], self.scale_precision[l])
            elif pd.noise_region_specific is False:
                self.scale_precision[l] = stats.ard_mean[l]/spectral_density[l] + \
                                          stats.noise_mean * np.sum(phi_x[l]*phi_x[l], axis=0)
                self.scale_mean_zeta[l] = np.divide(stats.noise_mean, self.scale_precision[l])
            else:
                raise TypeError('unsupported condition for noise_region_specific.')
            if pd.bias_region_specific is True:
                bias_mean = stats.bias_mean[l]
            elif pd.bias_region_specific is False:
                bias_mean = stats.bias_mean
            else:
                raise TypeError('unsupported condition for bias_region_specific.')
            self.scale_mean_y_tilde[l] = self._get_y_tilde(y_mean=y_mean[l],
                                                           phi_x=phi_x[l],
                                                           scale_axis_mean=stats.scale_axis_mean[l],
                                                           bias_mean=bias_mean,
                                                           f_bar=stats.latent_f_mean[l])

    def _get_y_tilde(self, y_mean, phi_x, scale_axis_mean, bias_mean, f_bar):
        y = y_mean
        y_tilde = np.zeros((self.dy, self.n_basis))
        for i in range(self.n_basis):
            sub_range = list(range(self.n_basis))
            del(sub_range[i])
            penalty_term = self._get_penalty_mean(sub_range, scale_axis_mean, phi_x)
            y_tilde_bar = f_bar + bias_mean + penalty_term
            y_tilde[:, i] = np.sum(np.tile(phi_x[:, i], (self.dy, 1)).T *
                                   (y - y_tilde_bar), axis=0)
        return y_tilde

    def _get_penalty_mean(self, subrange, scale_axis_mean, phi_x):
        n_samps = phi_x.shape[0]
        penalty = np.zeros((n_samps, self.dy))
        for k in subrange:
            penalty += scale_axis_mean[:, k] * np.tile(phi_x[:, k], (self.dy, 1)).T
        return penalty

    #   BIAS
    def update_bias_given_noise(self, y_mean, phi_x, prior, stats):
        if self.bias_region_specific is True:
            for l in range(self.n_regions):
                phi_x_l = phi_x[l]
                y_l = y_mean[l]
                n_samps = phi_x_l.shape[0]
                self.bias_normal_precision[l] = prior.bias_normal_precision[l] + n_samps
                sum_term = np.zeros_like(y_l)
                for i in range(self.n_basis):
                    sum_term += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
                self.bias_normal_mean[l] = np.power(self.bias_normal_precision[l], -1) * \
                    (prior.bias_normal_mean[l]*prior.bias_normal_precision[l] +
                     np.sum(y_l - sum_term - stats.latent_f_mean[l], axis=0))
        elif prior.bias_region_specific is False:
            n_samps_all_regions = 0
            for l in range(self.n_regions):
                n_samps_all_regions += phi_x[l].shape[0]
            self.bias_normal_precision = prior.bias_normal_precision + n_samps_all_regions
            sum_term = np.zeros(self.dy)
            for l in range(self.n_regions):
                phi_x_l = phi_x[l]
                y_l = y_mean[l]
                sum_term_l = np.zeros_like(y_l)
                for i in range(self.n_basis):
                    sum_term_l += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
                sum_term += np.sum(y_l - sum_term_l - stats.latent_f_mean[l], axis=0)
            self.bias_normal_mean = np.power(self.bias_normal_precision, -1) * \
                (prior.bias_normal_mean*prior.bias_normal_precision + sum_term)
        else:
            raise TypeError('Unknown condition for bias_region_specific.')

    #   NOISE
    def update_noise(self, y_mean, y_var, phi_x, prior, posterior, stats):
        if (self.noise_region_specific is True) and (self.bias_region_specific is True):
            self._update_regional_noise_regional_bias(y_mean, y_var,
                                                      phi_x, prior, posterior, stats)

        elif (self.noise_region_specific is True) and (self.bias_region_specific is False):
            self._update_regional_noise_shared_bias(y_mean, y_var,
                                                    phi_x, prior, posterior, stats)

        elif (self.noise_region_specific is False) and (self.bias_region_specific is True):
            self._update_shared_noise_regional_bias(y_mean, y_var,
                                                    phi_x, prior, posterior, stats)

        elif (self.noise_region_specific is False) and (self.bias_region_specific is False):
            self._update_shared_noise_shared_bias(y_mean, y_var,
                                                  phi_x, prior, posterior, stats)
        else:
            raise TypeError('Unknown conditions.')

    def _update_regional_noise_regional_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            self.noise_gamma_shape[l] = prior.noise_gamma_shape[l] + 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_l = y_var[l] * n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            term3 = prior.bias_normal_precision[l] * np.power(norm(prior.bias_normal_mean[l]), 2)
            term4 = posterior.bias_normal_precision[l]*np.power(norm(posterior.bias_normal_mean[l]), 2)
            self.noise_gamma_scale[l] = prior.noise_gamma_scale[l] +\
                0.5*(term3 - term4 + mean_term_l + var_f_l + var_au_l + y_var_l)

    def _update_regional_noise_shared_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        term3 = prior.bias_normal_precision * np.power(norm(prior.bias_normal_mean), 2)
        term4 = posterior.bias_normal_precision*np.power(norm(posterior.bias_normal_mean), 2)
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            self.noise_gamma_shape[l] = prior.noise_gamma_shape[l] + 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_l = y_var[l]
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            self.noise_gamma_scale[l] = prior.noise_gamma_scale[l] + \
                0.5*(term3 - term4 + mean_term_l + var_f_l + var_au_l + y_var_l)

    def _update_shared_noise_regional_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        noise_gamma_shape_update = 0
        noise_gamma_scale_update = 0
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            noise_gamma_shape_update += 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_l = y_var[l] * n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            term3 = prior.bias_normal_precision[l] * np.power(norm(prior.bias_normal_mean[l]), 2)
            term4 = posterior.bias_normal_precision[l]*np.power(norm(posterior.bias_normal_mean[l]), 2)
            noise_gamma_scale_update += 0.5*(term3 - term4 + mean_term_l + var_f_l + var_au_l + y_var_l)
        self.noise_gamma_shape = prior.noise_gamma_shape + noise_gamma_shape_update
        self.noise_gamma_scale = prior.noise_gamma_scale + noise_gamma_scale_update

    def _update_shared_noise_shared_bias(self, y_mean, y_var, phi_x, prior, posterior, stats):
        term3 = prior.bias_normal_precision * np.power(norm(prior.bias_normal_mean), 2)
        term4 = posterior.bias_normal_precision*np.power(norm(posterior.bias_normal_mean), 2)
        mean_term = 0
        var_f = 0
        var_au = 0
        y_var_sum = 0
        noise_gamma_shape_update = 0
        for l in range(self.n_regions):
            phi_x_l = phi_x[l]
            n_samps = phi_x_l.shape[0]
            noise_gamma_shape_update += 0.5*self.dy*n_samps
            y_l = y_mean[l]
            y_var_sum += y_var[l]*n_samps
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term += np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f += np.sum(stats.latent_f_var[l])
            var_au += np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
        noise_gamma_scale_update = 0.5*(term3 - term4 + mean_term + var_f + var_au + y_var_sum)
        self.noise_gamma_shape = prior.noise_gamma_shape + noise_gamma_shape_update
        self.noise_gamma_scale = prior.noise_gamma_scale + noise_gamma_scale_update


#   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:
#   :.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:..:.:


class SharedPosterior(object):
    def __init__(self, prior):
        self.axis_bingham_b = prior.axis_bingham_b.copy()
        self.axis_bingham_kappa = prior.axis_bingham_kappa.copy()
        self.axis_bingham_rho = prior.axis_bingham_rho.copy()
        self.axis_bingham_axes = prior.axis_bingham_axes.copy()
        self.axis_bingham_log_const = prior.axis_bingham_log_const.copy()

        self.ard_gamma_shape = np.copy(prior.ard_gamma_shape)
        self.ard_gamma_scale = np.copy(prior.ard_gamma_scale)

        self.dy = prior.dy
        self.n_basis = prior.n_basis

    #   BASIS AXIS
    def update_axis(self, prior, posterior, stats, shared_stats):
        n_regions = posterior.n_regions
        n_basis = self.n_basis
        dy = self.dy
        axis_bingham_b_prime = prior.axis_bingham_b
        for i in range(n_basis):
            sum_b = np.zeros((dy, dy))
            for k in range(n_basis):
                sum_b += shared_stats.omega[i, k]*axis_bingham_b_prime[k, :, :]
            sum_term = np.zeros((dy, dy))
            for l in range(n_regions):
                zeta_l = posterior.scale_mean_zeta[l][i]
                y_tilde_il = posterior.scale_mean_y_tilde[l][:, i]
                yy_tilde_zeta_l = zeta_l*np.outer(y_tilde_il, y_tilde_il)
                if posterior.noise_region_specific is True:
                    noise_mean = stats.noise_mean[l]
                elif posterior.noise_region_specific is False:
                    noise_mean = stats.noise_mean
                else:
                    raise TypeError('unsupported condition for noise_region_specific.')
                sum_term += (0.5 * noise_mean * yy_tilde_zeta_l)
            b = sum_b + sum_term
            sanity_check = SanityCheck()
            if sanity_check.isPD(b) is True:
                self.axis_bingham_b[i, :, :] = b
            else:
                self.axis_bingham_b[i, :, :] = sanity_check.nearestPD(b)
            bingham_pd = Bingham(self.axis_bingham_b[i, :, :])
            bingham_pd.kappa = np.real(bingham_pd.kappa)
            bingham_pd.kappa[bingham_pd.kappa < 0] = 0.
            self.axis_bingham_kappa[i, :] = bingham_pd.kappa
            self.axis_bingham_axes[i, :, :] = np.real(bingham_pd.axes)
            self.axis_bingham_rho[i, :] = np.real(bingham_pd.rho)
            self.axis_bingham_log_const[i] = np.real(bingham_pd.log_const)

    #   ARD
    def update_ard(self, prior, stats, shared_stats, spectral_density):
        for i in range(self.n_basis):
            shape_prime = prior.ard_gamma_shape
            scale_prime = prior.ard_gamma_scale
            self.ard_gamma_shape[i] = np.sum(shared_stats.omega[i, :]*shape_prime) + 0.5*stats.n_regions
            beta_term2 = 0
            for l in range(stats.n_regions):
                beta_term2 += stats.scale_moment2[l][i]/spectral_density[l][i]
            self.ard_gamma_scale[i] = np.sum(shared_stats.omega[i, :]*scale_prime) + 0.5*beta_term2
