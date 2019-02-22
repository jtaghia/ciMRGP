import numpy as np
from scipy import optimize
import warnings


class BasisInterval(object):

    def __init__(self, use_prior=True, opt_interval_factor=(1., 1.2)):
        self.basis_interval = None
        self.basis_function_obj = None
        self.spectral_density_obj = None
        self.use_prior = use_prior
        self.opt_interval_factor = opt_interval_factor

    def max_input_range_by_factor_of(self, inputs, factor):
        return factor * np.max(np.abs(inputs), axis=0)

    def learn(self, x, y, stats, shared_stats, phi_x_penalty, lambda_penalty, basis_function,
              spectral_density):
        self.basis_function_obj = basis_function
        self.spectral_density_obj = spectral_density
        if shared_stats is None:
            ard_mean_ = stats.ard_mean
        else:
            ard_mean = shared_stats.ard_mean
        basis_interval_opt = []
        for l in range(stats.n_regions):
            if stats.noise_region_specific is True:
                noise_mean = stats.noise_mean[l]
            elif stats.noise_region_specific is False:
                noise_mean = stats.noise_mean
            else:
                raise ValueError('not a valid condition')
            if stats.bias_region_specific is True:
                bias_mean = stats.bias_mean[l]
            elif stats.bias_region_specific is False:
                bias_mean = stats.bias_mean
            else:
                raise ValueError('not a valid condition')

            axis_mean_l = stats.scale_axis_mean[l]
            scale_moment2_l = stats.scale_moment2[l]
            var_au_l = stats.scale_axis_central_moment2[l]
            f_bar_l = stats.latent_f_mean[l]
            x_l = x[l]
            y_l = y[l]
            phi_x_penalty_l = phi_x_penalty[l]
            lambda_penalty_l = lambda_penalty[l]
            if shared_stats is None:
                ard_mean = ard_mean_[l]
            bi_opt_l = self._learn(x_l, y_l, noise_mean, bias_mean, ard_mean,
                                   axis_mean_l, scale_moment2_l, var_au_l, f_bar_l,
                                   phi_x_penalty_l, lambda_penalty_l)
            basis_interval_opt.append(bi_opt_l)
        return basis_interval_opt

    def _get_axis_mean_penalty(self, subrange, scale_axis_mean, phi_x):
        n_samps = phi_x.shape[0]
        dy = scale_axis_mean.shape[0]
        penalty = np.zeros((n_samps, dy))
        for k in subrange:
            penalty += scale_axis_mean[:, k] * np.tile(phi_x[:, k], (dy, 1)).T
        return penalty

    def _learn(self, x, y, noise_mean, bias_mean, ard_mean,
               axis_mean, scale_moment2, var_au_l, f_bar_l,
               phi_x_penalty, lambda_penalty):

        dx = x.shape[1]
        dy, self.n_basis = axis_mean.shape
        self.n_samps = y.shape[0]
        opt_basis_interval = np.zeros(dx)
        for p in range(dx):
            lambda_penalty_p = lambda_penalty[:, p]
            x_p = np.atleast_2d(x[:, p]).T
            phi_penalty_p = phi_x_penalty[:, :, p]
            interval_low = np.max(abs(x_p)) * self.opt_interval_factor[0]
            interval_high = min(self.n_basis, interval_low*self.opt_interval_factor[1])
            if interval_high < interval_low:
                warnings.warn(" ----> WARNING: \n Number of basis functions is less than the input range. "
                              "The Problem seems to be too large of an input range."
                              "The result might be suboptimal. Try changing the low interval factor to "
                              "values smaller than 1. It might be also helpful to normalize the inputs.")
                interval_high = interval_low*self.opt_interval_factor[1]
            opt_basis_interval[p] = optimize.fminbound(self.h, interval_low, interval_high,
                                                       args=(x_p, y, noise_mean, ard_mean, scale_moment2,
                                                             axis_mean, var_au_l,
                                                             lambda_penalty_p, phi_penalty_p,
                                                             bias_mean, f_bar_l),
                                                       full_output=0)

        return opt_basis_interval

    def h(self, interval,
          x, y, noise_mean, ard_mean, scale_moment2, axis_mean, var_au_l,
          lambda_penalty, phi_penalty_p, bias_mean, f_bar_l):

        lambda_ = np.zeros(self.n_basis)
        phi_x = np.ones((self.n_samps, self.n_basis))
        for i in range(self.n_basis):
            basis_id = i + 1
            phi_x_full, lambda_full = self.basis_function_obj.get_eigenpairs(x=x, basis_id=basis_id,
                                                                             basis_interval=[interval],
                                                                             per_dimension=True)
            lambda_[i] = np.sum(lambda_full)
            phi_x[:, i] = np.prod(phi_x_full, axis=1)

        if self.use_prior is True:
            spectral_density_ = np.zeros(self.n_basis)
            for i in range(self.n_basis):
                spectral_density_[i] = self.spectral_density_obj.spectral(np.sqrt(lambda_[i] +
                                                                                  lambda_penalty[i]))
        else:
            spectral_density_ = np.ones(self.n_basis)

        dy = axis_mean.shape[0]
        ll_term = 0
        for i in range(self.n_basis):
            term1 = sum(np.power(np.linalg.norm(axis_mean[:, i]), 2) *\
                        np.power(phi_penalty_p[:, i]*phi_x[:, i], 2))
            phi_temp = np.tile(phi_penalty_p[:, i]*phi_x[:, i], (dy, 1)).T
            term2 = sum(np.inner((bias_mean + f_bar_l) * phi_temp, axis_mean[:, i].T))
            term3 = sum(np.inner(y * phi_temp, axis_mean[:, i].T))
            term4 = sum(var_au_l[i] * np.power(phi_penalty_p[:, i]*phi_x[:, i], 2))
            ll_term += 2*term1 + 4*term2 - 2*term3 + term4
        ll_term *= -0.5*noise_mean
        if self.use_prior is False:
            h_objective = ll_term
        else:
            prior_term = -0.5*np.sum(np.log(spectral_density_) - \
                                     0.5 * (ard_mean * scale_moment2)/spectral_density_)
            h_objective = ll_term + prior_term

        return - h_objective
