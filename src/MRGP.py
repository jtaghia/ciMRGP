from copy import deepcopy
from tqdm import *

from BasisInterval import BasisInterval
from Inputs import Inputs
from Priors import *
from Posteriors import *
from Stats import *
from LatentOutputs import LatentOutputs


EPSILON = 1e-100


class MultiResolutionGaussianProcess(object):
    def __init__(self, train_xy,
                 n_basis,
                 index_set_obj,
                 basis_function_obj,
                 spectral_density_obj=None,
                 basis_interval_obj=None,
                 interval_factor=1,
                 adaptive_inputs=False,
                 standard_normalized_inputs=True,
                 axis_resolution_specific=False,
                 ard_resolution_specific=False,
                 noise_region_specific=True,
                 bias_region_specific=True,
                 noninformative_initialization=True,
                 snr_ratio=None,
                 full_x=None,
                 input_model=None,
                 forced_independence=False,
                 verbose=False):

        self.verbose = verbose
        self.forced_independence = forced_independence
        if forced_independence is True:
            self.axis_resolution_specific = True
            self.ard_resolution_specific = True
            if self.verbose is True:
                print("*** All GPs are forced to be independent *** ")
        else:
            if (axis_resolution_specific is False) and (ard_resolution_specific is False):
                self.axis_resolution_specific = False
                self.ard_resolution_specific = False
                if self.verbose is True:
                    print(" \n *** GPs are conditionally independent given the basis axes *** \n ")
            else:
                raise TypeError("not yet supported")

        self.adaptive_inputs = adaptive_inputs
        self.standard_normalized_inputs = standard_normalized_inputs

        self.noise_region_specific = noise_region_specific
        self.bias_region_specific = bias_region_specific

        self.n_layers = index_set_obj.get_n_resolutions() + 1
        self.index_set_obj = index_set_obj
        self.n_basis = n_basis
        x_train = train_xy[0]
        y_train = train_xy[1]
        self.observations = y_train
        self.dy = y_train.shape[1]
        if self.dy < 2:
            raise ValueError('output dimension must be greater than 1')

        #  Normalize if necessary
        x_train, self.full_x, self.mean_x_train, self.std_x_train = self._normalize_inputs(x_train, full_x)

        if isinstance(spectral_density_obj, list) is False:
            self.spectral_density_obj = [spectral_density_obj] * self.n_layers
        elif isinstance(spectral_density_obj, list) is True:
            if len(spectral_density_obj) != self.n_layers:
                raise ValueError('spectral_density_obj must be a list of the same length as the number of '
                                 'resolutions + 1')
            self.spectral_density_obj = spectral_density_obj
        else:
            raise ValueError('spectral_density_obj not recognized')
        if isinstance(basis_function_obj, list) is False:
            self.basis_function_obj = [basis_function_obj] * self.n_layers
        elif isinstance(basis_function_obj, list) is True:
            if len(basis_function_obj) != self.n_layers:
                raise ValueError('basis_function_obj must be a list of the same length as the number of '
                                 'resolutions + 1')
            self.basis_function_obj = basis_function_obj
        else:
            raise ValueError('basis_function_obj not recognized')

        self.use_prior = []
        for j in range(self.n_layers):
            if self.spectral_density_obj[j] is None:
                self.use_prior.append(False)
            else:
                self.use_prior.append(True)

        #   Basis interval class
        if isinstance(interval_factor, list) is False:
            self.interval_factor = [interval_factor] * self.n_layers
        elif isinstance(interval_factor, list) is True:
            if len(interval_factor) != self.n_layers:
                raise ValueError('interval_factor must be a list of the same length as the number of '
                                 'resolutions + 1')
            self.interval_factor = interval_factor
        else:
            raise ValueError('interval_factor not recognized')

        if forced_independence is True:
            basis_interval_obj = None

        self.basis_interval_obj = []
        if basis_interval_obj is None:
            self.adaptive_basis_intervals = False
            for j in range(self.n_layers):
                basis_interval_obj_j = BasisInterval()
                self.basis_interval_obj.append(basis_interval_obj_j)
        else:
            self.adaptive_basis_intervals = True
            if isinstance(basis_interval_obj, list) is False:
                self.basis_interval_obj = [basis_interval_obj] * self.n_layers
            else:
                if len(basis_interval_obj) != self.n_layers:
                    raise ValueError('basis_interval_obj must be a list of the same length as the number of '
                                     'resolutions + 1')
                else:
                    self.basis_interval_obj = basis_interval_obj

        self.input_obj = Inputs(x=x_train, index_set=index_set_obj, learn_inputs=self.adaptive_inputs,
                                full_x=self.full_x, input_model=input_model)

        self.dx = self.input_obj.x.shape[1]
        self.n_regions = []
        for j in range(self.n_layers):
            self.n_regions.append(len(index_set_obj.index_set[j]))

        #  compute initial basis functions and basis intervals for training
        self.train_basis_intervals = []
        self.x = []
        self.n_samps = []
        for j in range(self.n_layers):
            x_j = []
            train_basis_intervals_j = []
            n_samps_j = []
            for l in range(self.n_regions[j]):
                x_jl = self.input_obj.get_inputs(resolution=j, region=l)
                interval_jl = \
                    self.basis_interval_obj[j].max_input_range_by_factor_of(inputs=x_jl,
                                                                            factor=self.interval_factor[j])
                train_basis_intervals_j.append(interval_jl)
                x_j.append(x_jl)
                n_samps_j.append(x_jl.shape[0])
            self.x.append(x_j)
            self.train_basis_intervals.append(train_basis_intervals_j)
            self.n_samps.append(n_samps_j)

        #  initialize basis functions and spectral densities
        self.phi_x = []
        self.phi_x_penalty = []
        self.lambda_ = []
        self.lambda_penalty = []
        self.spectral_density_prior = []
        for j in range(self.n_layers):
            self.phi_x.append([] * self.n_regions[j])
            self.phi_x_penalty.append([] * self.n_regions[j])
            self.lambda_penalty.append([] * self.n_regions[j])
            self.lambda_.append([] * self.n_regions[j])
            self.spectral_density_prior.append([] * self.n_regions[j])

        #   TODO: check this carefully
        for j in range(self.n_layers):
            self._update_basis_functions(resolution=j)
            self._update_spectral_density(resolution=j)

        sf = []
        for j in range(self.n_layers):
            if self.spectral_density_obj[j] is None:
                sf.append(1.)
            else:
                sf.append(self.spectral_density_obj[j].sf)

        #  initialize the prior object for all resolutions
        if (self.axis_resolution_specific is False) and (self.ard_resolution_specific is False):
            self.shared_prior = SharedPrior(self.n_basis, self.dy)
            self.shared_prior.basis_axis(noninformative=noninformative_initialization)
            self.shared_prior.ard(prior_influence=np.mean(sf),
                                  noninformative=noninformative_initialization)

            self.prior_obj = []
            for j in range(self.n_layers):
                prior_j = Prior(self.n_basis, self.dy, self.n_regions[j])
                prior_j.basis_axis_scale(spectral_density=self.spectral_density_prior[j],
                                         noninformative=noninformative_initialization)
                prior_j.bias(region_specific=bias_region_specific,
                             noninformative=noninformative_initialization)
                if j == 0:
                    if snr_ratio is not None:
                        noise_var = self._compute_initial_noise_var_from_snr(y=y_train, snr_ratio=snr_ratio)
                    else:
                        noise_var = None
                else:
                    noise_var = None
                prior_j.noise(noise_var=noise_var, region_specific=noise_region_specific,
                              noninformative=noninformative_initialization)
                self.prior_obj.append(prior_j)
        elif (self.axis_resolution_specific is True) and (self.ard_resolution_specific is True):
            self.prior_obj = []
            for j in range(self.n_layers):
                prior_j = IndependentPrior(self.n_basis, self.dy, self.n_regions[j])
                prior_j.basis_axis(noninformative=noninformative_initialization)
                prior_j.ard(prior_influence=np.mean(sf), noninformative=noninformative_initialization)
                prior_j.basis_axis_scale(spectral_density=self.spectral_density_prior[j],
                                         noninformative=noninformative_initialization)
                prior_j.bias(region_specific=bias_region_specific,
                             noninformative=noninformative_initialization)
                if j == 0:
                    if snr_ratio is not None:
                        noise_var = self._compute_initial_noise_var_from_snr(y=y_train, snr_ratio=snr_ratio)
                    else:
                        noise_var = None
                else:
                    noise_var = None
                prior_j.noise(noise_var=noise_var, region_specific=noise_region_specific,
                              noninformative=noninformative_initialization)
                self.prior_obj.append(prior_j)
        else:
            raise TypeError('not supported')

        #  initialize the posterior object for all resolutions
        if (self.axis_resolution_specific is False) and (self.ard_resolution_specific is False):
            self.shared_posterior = SharedPosterior(prior=self.shared_prior)

            self.posterior_obj = []
            for j in range(self.n_layers):
                post_j = Posterior(prior=self.prior_obj[j])
                self.posterior_obj.append(post_j)
        elif (self.axis_resolution_specific is True) and (self.ard_resolution_specific is True):
            self.posterior_obj = []
            for j in range(self.n_layers):
                post_j = IndependentPosterior(prior=self.prior_obj[j])
                self.posterior_obj.append(post_j)
        else:
            raise TypeError('not yet implemented')

        #  initialize the stats object for all resolutions
        if (self.axis_resolution_specific is False) and (self.ard_resolution_specific is False):
            self.shared_stats = SharedStats(posterior=self.shared_posterior)
            self.stats_obj = []
            for j in range(self.n_layers):
                stats_j = Stats(posterior=self.posterior_obj[j])
                stats_j.initialize_latent_functions(self.n_samps[j])
                self.stats_obj.append(stats_j)
        elif (self.axis_resolution_specific is True) and (self.ard_resolution_specific is True):
            self.stats_obj = []
            for j in range(self.n_layers):
                stats_j = IndependentStats(posterior=self.posterior_obj[j])
                stats_j.initialize_latent_functions(self.n_samps[j])
                self.stats_obj.append(stats_j)
        else:
            raise TypeError('not yet implemented')
        #  instantiate the output object
        self.output_obj = LatentOutputs()
        self.y_mean = []
        self.y_var = []
        for j in range(self.n_layers):
            y_mean_j = []
            y_var_j = []
            for l in range(self.n_regions[j]):
                y_mean_j.append([])
                y_var_j.append([])
            self.y_mean.append(y_mean_j)
            self.y_var.append(y_var_j)

        self.lower_bound_layer = []
        for j in range(self.n_layers):
            self.lower_bound_layer.append([])
        self.lower_bound = []

    def _normalize_inputs(self, x_train, full_x):
        if full_x is None:
            x = x_train
        else:
            x = full_x
        if self.standard_normalized_inputs is True:
            std_x_train = np.std(x, 0)
            std_x_train[std_x_train == 0] = 1
            mean_x_train = np.mean(x, 0)
            x_train = (x_train - np.full(x_train.shape, mean_x_train)) / \
                np.full(x_train.shape, std_x_train)
            if full_x is not None:
                full_x = (full_x - np.full(full_x.shape, mean_x_train)) / \
                    np.full(full_x.shape, std_x_train)
        else:
            mean_x_train = None
            std_x_train = None
        return x_train, full_x, mean_x_train, std_x_train

    def _get_prior_spectral_density(self, lambda_jl, resolution):
        j = resolution
        spectral_density_jl = np.zeros(self.n_basis)
        for i in range(self.n_basis):
            # TODO: here we have two options: To assume S(lambda) or S(sqrt(lambda)). Which one is it?
            spectral_density_jl[i] = self.spectral_density_obj[j].spectral(np.sqrt(lambda_jl[i]))
        return spectral_density_jl

    def _update_basis_functions(self, resolution):
        j = resolution
        phi_x_j = []
        lambda_j = []
        phi_x_penalty_j = []
        lambda_penalty_j = []
        for l in range(self.n_regions[j]):
            x_jl = self.x[j][l]
            basis_interval_jl = self.train_basis_intervals[j][l]
            phi_x_jl, lambda_jl, phi_x_penalty_jl, lambda_penalty_jl = \
                self._learn_basis_functions_jl(x=x_jl, interval=basis_interval_jl, resolution=j)
            phi_x_j.append(phi_x_jl)
            lambda_j.append(lambda_jl)
            phi_x_penalty_j.append(phi_x_penalty_jl)
            lambda_penalty_j.append(lambda_penalty_jl)
        self.phi_x[j] = phi_x_j
        self.lambda_[j] = lambda_j
        self.lambda_penalty[j] = lambda_penalty_j
        self.phi_x_penalty[j] = phi_x_penalty_j

    def _update_spectral_density(self, resolution):
        j = resolution
        spectral_density_j = []
        if self.spectral_density_obj[j] is None:
            for l in range(self.n_regions[j]):
                spectral_density_j.append(np.ones(self.n_basis))
        else:
            for l in range(self.n_regions[j]):
                spectral_density_j.append(self._get_prior_spectral_density(lambda_jl=self.lambda_[j][l],
                                                                           resolution=j))
        self.spectral_density_prior[j] = spectral_density_j

    def _learn_basis_functions_jl(self, x, interval, resolution):
        j = resolution
        n_samps = x.shape[0]
        phi_x = np.ones((n_samps, self.n_basis))
        lambda_ = np.zeros(self.n_basis)
        phi_x_penalty = np.ones((n_samps, self.n_basis, self.dx))
        lambda_penalty = np.zeros((self.n_basis, self.dx))
        for i in range(self.n_basis):
            basis_id = i + 1
            phi_x_full, lambda_full = self.basis_function_obj[j].get_eigenpairs(x=x, basis_id=basis_id,
                                                                                basis_interval=interval,
                                                                                per_dimension=True)
            lambda_[i] = np.sum(lambda_full)
            phi_x[:, i] = np.prod(phi_x_full, axis=1)
            if self.dx > 1:
                for dim in range(self.dx):
                    sub_range_dx = list(range(self.dx))
                    del(sub_range_dx[dim])
                    phi_x_penalty[:, i, dim] = np.prod(phi_x_full[:, sub_range_dx], axis=1)
                    lambda_penalty[i, dim] = np.sum(lambda_full[sub_range_dx])
        return phi_x, lambda_, phi_x_penalty, lambda_penalty

    @property
    def get_posterior(self):
        return self.posterior_obj

    @property
    def get_stats(self):
        return self.stats_obj

    def fit(self, n_iter=1, tol=1e-3, min_iter=10):
        if tol is None:
            self._fit_iter(n_iter)
        else:
            self._fit_tol(tol, n_iter, min_iter=min_iter)

    def _fit_tol(self, tol, n_iter, min_iter):
        if n_iter < min_iter:
            min_iter = n_iter
        for iter_ in range(n_iter):
            iter_ += 1
            if (self.axis_resolution_specific is False) and (self.ard_resolution_specific is False):
                prime_shared_posterior = self.shared_posterior
                self._fit()
                lower_bound, lower_bound_layer =\
                    self._compute_lower_bound(prime_shared_posterior=prime_shared_posterior)
                self.lower_bound.append(lower_bound)
                for j in range(self.n_layers):
                    self.lower_bound_layer[j].append(lower_bound_layer[j])
                if iter_ > min_iter:
                    delta_elbo = self.lower_bound[-1] - self.lower_bound[-2]
                    delta_elbo_layer = []
                    for j in range(self.n_layers):
                        delta_elbo_layer.append(self.lower_bound_layer[j][-1] - self.lower_bound_layer[j][-2])
                        if self.verbose is True:
                            print("    layer" + str(j) + ":   ELBO " + "%.f2 ... dELBO %.2f"
                                  % (self.lower_bound_layer[j][-1], delta_elbo_layer[j]))
                    if self.verbose is True:
                        print("\nTotal ELBO: %.4f ... dELBO %.4f" % (self.lower_bound[-1], delta_elbo))
                    if abs(delta_elbo_layer[0]) < abs(tol):
                        if self.verbose is True:
                            print("converged: dELBO is smaller than %s" % str(tol))
                        break
            elif (self.axis_resolution_specific is True) and (self.ard_resolution_specific is True):
                self._independent_fit()
            else:
                raise TypeError('not yet supported')

    def _fit_iter(self, n_iter=1):
        for iter_ in tqdm(range(n_iter), desc="final learning"):
            if (self.axis_resolution_specific is False) and (self.ard_resolution_specific is False):
                self._fit()
            elif (self.axis_resolution_specific is True) and (self.ard_resolution_specific is True):
                self._independent_fit()
            else:
                raise TypeError('not yet supported')

    def _compute_lower_bound(self, prime_shared_posterior):
        ll = []
        for j in range(self.n_layers):
            data_likelihood = self._data_likelihood(res=j)
            ll_scale_given_axis = self._ll_scale_given_axis(res=j)
            ll_axis = self._ll_axis(res=j, prime_shared_posterior=prime_shared_posterior)
            ll_ard = self._ll_ard(res=j, prime_shared_posterior=prime_shared_posterior)
            ll_bias = self._ll_bias(res=j)
            ll_noise = self._ll_noise(res=j)
            ll.append(data_likelihood + ll_scale_given_axis + ll_axis + ll_ard + ll_bias + ll_noise)
        return np.sum(ll), ll

    def _ll_noise(self, res):
        j = res
        log_p = 0
        log_q = 0
        for l in range(self.n_regions[j]):
            if self.noise_region_specific is True:
                c0 = self.prior_obj[j].noise_gamma_shape[l]
                d0 = self.prior_obj[j].noise_gamma_scale[l]
                c = self.posterior_obj[j].noise_gamma_shape[l]
                d = self.posterior_obj[j].noise_gamma_scale[l]
                noise_log_mean = self.stats_obj[j].noise_log_mean[l]
                noise_mean = self.stats_obj[j].noise_mean[l]
            else:
                c0 = self.prior_obj[j].noise_gamma_shape
                d0 = self.prior_obj[j].noise_gamma_scale
                c = self.posterior_obj[j].noise_gamma_shape
                d = self.posterior_obj[j].noise_gamma_scale
                noise_log_mean = self.stats_obj[j].noise_log_mean
                noise_mean = self.stats_obj[j].noise_mean
            log_p += c0*np.log(d0) - gammaln(c0) + (c0-1)*noise_log_mean - d0*noise_mean
            log_q += c*np.log(d) - gammaln(c) + (c-1)*noise_log_mean - d*noise_mean
        return log_p - log_q

    def _ll_bias(self, res):
        j = res
        log_p = 0
        log_q = 0
        for l in range(self.n_regions[j]):
            if self.noise_region_specific is True:
                noise_log_mean = self.stats_obj[j].noise_log_mean[l]
                noise_mean = self.stats_obj[j].noise_mean[l]
            else:
                noise_log_mean = self.stats_obj[j].noise_log_mean
                noise_mean = self.stats_obj[j].noise_mean
            if self.bias_region_specific is True:
                w = self.posterior_obj[j].bias_normal_mean[l]
                tau = self.posterior_obj[j].bias_normal_precision[l]
                w0 = self.prior_obj[j].bias_normal_mean[l]
                tau0 = self.prior_obj[j].bias_normal_precision[l]
            else:
                w = self.posterior_obj[j].bias_normal_mean
                tau = self.posterior_obj[j].bias_normal_precision
                w0 = self.prior_obj[j].bias_normal_mean
                tau0 = self.prior_obj[j].bias_normal_precision
            const = 0.5*self.dy*(np.log(tau0) + noise_log_mean - np.log(2*np.pi))
            term1 = np.divide(1, tau*noise_mean) + np.dot(w, w) - 2*np.dot(w, w0) + np.dot(w0, w0)
            exp_term = 0.5*tau0*noise_mean*term1
            log_p += const + exp_term
            log_q += 0.5*self.dy*(np.log(tau) + noise_log_mean - np.log(2*np.pi)) - 0.5
        return log_p - log_q

    def _ll_ard(self, res, prime_shared_posterior):
        j = res
        if j == 0:
            prime_gqd = self.shared_prior
        else:
            prime_gqd = prime_shared_posterior
        gst = self.shared_stats
        gqd = self.shared_posterior
        n_basis = self.n_basis
        log_p = 0
        for i in range(n_basis):
            for k in range(n_basis):
                log_term = prime_gqd.ard_gamma_shape[k]*np.log(prime_gqd.ard_gamma_scale[k]) \
                           - gammaln(prime_gqd.ard_gamma_shape[k]) + \
                (prime_gqd.ard_gamma_shape[k]-1)*gst.ard_log_mean[i] - prime_gqd.ard_gamma_scale[k]*gst.ard_mean[i]
                log_p += gst.omega[i, k] * log_term
        log_q = 0
        for i in range(n_basis):
            log_q += gqd.ard_gamma_shape[i]*np.log(gqd.ard_gamma_scale[i]) - gammaln(gqd.ard_gamma_shape[i]) + \
                (gqd.ard_gamma_shape[i]-1)*gst.ard_log_mean[i] - gqd.ard_gamma_scale[i]*gst.ard_mean[i]
        return log_p - log_q

    def _ll_axis(self, res, prime_shared_posterior):
        j = res
        gst = self.shared_stats
        gqd = self.shared_posterior
        n_basis = self.n_basis
        if j == 0:
            prime_gqd = self.shared_prior
        else:
            prime_gqd = prime_shared_posterior
        log_p = 0
        for i in range(n_basis):
            for k in range(n_basis):
                log_term = -prime_gqd.axis_bingham_log_const[k] + \
                           np.trace(gst.axis_cov[i, :, :] * prime_gqd.axis_bingham_b[k, :, :])
                log_p += gst.omega[i, k] * log_term
        log_q = 0
        for i in range(n_basis):
            log_q += -gqd.axis_bingham_log_const[i] + \
                           np.trace(gst.axis_cov[i, :, :] * gqd.axis_bingham_b[i, :, :])
        return log_p - log_q

    def _ll_scale_given_axis(self, res):
        j = res
        gst = self.shared_stats
        st = self.stats_obj[j]
        qd = self.posterior_obj[j]
        spec_den = self.spectral_density_prior[j]
        log_p = 0.
        for l in range(self.n_regions[j]):
            log_p += np.sum(0.5*gst.ard_log_mean/spec_den[l] - 0.5*gst.ard_mean*st.scale_moment2[l]/spec_den[l])

        log_q = 0.
        for l in range(self.n_regions[j]):
            log_q += np.sum(0.5*np.log(qd.scale_precision[l, :]) - .5)
        return log_p - log_q

    def _data_likelihood(self, res):
        j = res
        phi_x = self.phi_x[j]
        y_var = self.y_var[j]
        y_mean = self.y_mean[j]
        n_samps = self.n_samps[j]
        stats = self.stats_obj[j]
        log_ll_j = 0
        const = 0
        for l in range(self.n_regions[j]):
            phi_x_l = phi_x[l]
            y_l = y_mean[l]
            # TODO: is it zero or should be there
            y_var_l = y_var[l] * n_samps[l]
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            if self.bias_region_specific is True:
                bias_mean = stats.bias_mean[l]
                bias_var = stats.bias_var[l]
            else:
                bias_mean = stats.bias_mean
                bias_var = stats.bias_var
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l] - bias_mean, axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])

            log_ll_j += mean_term_l + var_f_l + var_au_l + bias_var + y_var_l
            if self.noise_region_specific is True:
                noise_log_mean = stats.noise_log_mean[l]
            else:
                noise_log_mean = stats.noise_log_mean
            const += 0.5*self.dy * (noise_log_mean - np.log(2*np.pi)) * n_samps[l]
        log_ll_j += const
        return log_ll_j

    def _fit(self):
        for j in range(self.n_layers):
            if j == 0:
                y_mean, y_var = self.output_obj.get_stats(observations=self.observations)
                previous_posterior = deepcopy(self.shared_prior)
            else:
                y_mean, y_var = self.output_obj.infer_stats(stats=self.stats_obj[j],
                                                            basis_functions=self.phi_x[j])
                # y_mean, y_var = self.output_obj.infer_point_stats(index_set=self.index_set_obj.index_set[j],
                #                                                   observations=self.observations)
                previous_posterior = deepcopy(self.shared_posterior)

            #  UPDATE SCALE GIVEN AXIS POSTERIOR
            self.posterior_obj[j].update_scale_given_axis(y_mean=y_mean,
                                                          phi_x=self.phi_x[j],
                                                          prior=self.prior_obj[j],
                                                          stats=self.stats_obj[j],
                                                          shared_stats=self.shared_stats,
                                                          spectral_density=self.spectral_density_prior[j])

            #  UPDATE AXIS
            self.shared_posterior.update_axis(prior=previous_posterior,
                                              posterior=self.posterior_obj[j],
                                              stats=self.stats_obj[j],
                                              shared_stats=self.shared_stats)

            #  UPDATE SCALE and AXIS STATS
            self.shared_stats.update_axis(posterior=self.shared_posterior)
            self.stats_obj[j].update_scale(posterior=self.posterior_obj[j],
                                           stats=self.shared_stats)

            #  UPDATE ARD POSTERIOR
            self.shared_posterior.update_ard(prior=previous_posterior,
                                             stats=self.stats_obj[j],
                                             shared_stats=self.shared_stats,
                                             spectral_density=self.spectral_density_prior[j])

            #   UPDATE ARD STATS
            self.shared_stats.update_ard(posterior=self.shared_posterior)

            #   UPDATE PERMUTATION ALIGNMENT STATS
            if (self.axis_resolution_specific is False) and (self.ard_resolution_specific is False):
                self.shared_stats.update_omega(prior=previous_posterior, stats=self.shared_stats)
            else:
                raise TypeError('not yet implemented...')

            #   UPDATE BIAS GIVEN NOISE POSTERIOR
            self.posterior_obj[j].update_bias_given_noise(y_mean=y_mean,
                                                          phi_x=self.phi_x[j],
                                                          prior=self.prior_obj[j],
                                                          stats=self.stats_obj[j])
            #   UPDATE NOISE
            self.posterior_obj[j].update_noise(y_mean=y_mean, y_var=y_var,
                                               phi_x=self.phi_x[j],
                                               prior=self.prior_obj[j],
                                               posterior=self.posterior_obj[j],
                                               stats=self.stats_obj[j])
            #   UPDATE NOISE and BIAS STATS
            self.stats_obj[j].update_bias(posterior=self.posterior_obj[j])
            self.stats_obj[j].update_noise(posterior=self.posterior_obj[j])

            if self.adaptive_basis_intervals is True:
                self.train_basis_intervals[j] = \
                    self.basis_interval_obj[j].learn(x=self.x[j], y=y_mean, stats=self.stats_obj[j],
                                                     shared_stats=self.shared_stats,
                                                     phi_x_penalty=self.phi_x_penalty[j],
                                                     lambda_penalty=self.lambda_penalty[j],
                                                     basis_function=self.basis_function_obj[j],
                                                     spectral_density=self.spectral_density_obj[j])
                self._update_basis_functions(resolution=j)
                self._update_spectral_density(resolution=j)

        #   UPDATE LATENT FUNCTION STATS
            next_resolution = j+1
            if next_resolution < self.n_layers:
                self.stats_obj[next_resolution].update_latent_functions(resolution=next_resolution,
                                                                        index_set=self.index_set_obj.index_set,
                                                                        stats=self.stats_obj,
                                                                        phi_x=self.phi_x)
            for l in range(self.n_regions[j]):
                self.y_mean[j][l] = y_mean[l]
                self.y_var[j][l] = y_var[l]

    def _independent_fit(self):
        for j in range(self.n_layers):
            if j == 0:
                y_mean, y_var = self.output_obj.get_stats(observations=self.observations)
            else:
                # y_mean, y_var = self.output_obj.infer_stats(stats=self.stats_obj[j],
                #                                             basis_functions=self.phi_x[j])
                y_mean, y_var = self.output_obj.infer_point_stats(index_set=self.index_set_obj.index_set[j],
                                                                  observations=self.observations)

            #  UPDATE SCALE GIVEN AXIS POSTERIOR
            self.posterior_obj[j].update_scale_given_axis(y_mean=y_mean,
                                                          phi_x=self.phi_x[j],
                                                          prior=self.prior_obj[j],
                                                          stats=self.stats_obj[j],
                                                          spectral_density=self.spectral_density_prior[j])

            #  UPDATE AXIS
            self.posterior_obj[j].update_axis(prior=self.prior_obj[j],
                                              posterior=self.posterior_obj[j],
                                              stats=self.stats_obj[j])

            #  UPDATE SCALE and AXIS STATS
            self.stats_obj[j].update_axis(posterior=self.posterior_obj[j])
            self.stats_obj[j].update_scale(posterior=self.posterior_obj[j], stats=self.stats_obj[j])

            #  UPDATE ARD POSTERIOR
            self.posterior_obj[j].update_ard(prior=self.prior_obj[j],
                                             stats=self.stats_obj[j],
                                             spectral_density=self.spectral_density_prior[j])

            #   UPDATE ARD STATS
            self.stats_obj[j].update_ard(posterior=self.posterior_obj[j])

            #   UPDATE PERMUTATION ALIGNMENT STATS
            if self.forced_independence is False:
                self.shared_stats.update_omega(prior=self.prior_obj, stats=self.shared_stats)

            #   UPDATE BIAS GIVEN NOISE POSTERIOR
            self.posterior_obj[j].update_bias_given_noise(y_mean=y_mean,
                                                          phi_x=self.phi_x[j],
                                                          prior=self.prior_obj[j],
                                                          stats=self.stats_obj[j])
            #   UPDATE NOISE
            self.posterior_obj[j].update_noise(y_mean=y_mean, y_var=y_var,
                                               phi_x=self.phi_x[j],
                                               prior=self.prior_obj[j],
                                               posterior=self.posterior_obj[j],
                                               stats=self.stats_obj[j])
            #   UPDATE NOISE and BIAS STATS
            self.stats_obj[j].update_bias(posterior=self.posterior_obj[j])
            self.stats_obj[j].update_noise(posterior=self.posterior_obj[j])

            if self.adaptive_basis_intervals is True:
                self.train_basis_intervals[j] = \
                    self.basis_interval_obj[j].learn(x=self.x[j], y=y_mean, stats=self.stats_obj[j],
                                                     shared_stats=None,
                                                     phi_x_penalty=self.phi_x_penalty[j],
                                                     lambda_penalty=self.lambda_penalty[j],
                                                     basis_function=self.basis_function_obj[j],
                                                     spectral_density=self.spectral_density_obj[j])
                self._update_basis_functions(resolution=j)
                self._update_spectral_density(resolution=j)

        #   UPDATE LATENT FUNCTION STATS
            next_resolution = j+1
            if next_resolution < self.n_layers:
                self.stats_obj[next_resolution].update_latent_functions(resolution=next_resolution,
                                                                        index_set=self.index_set_obj.index_set,
                                                                        stats=self.stats_obj,
                                                                        phi_x=self.phi_x)

    def _get_get_predicted_mean_global(self, test_x):
        #  every predictions are taken from resolution 0
        n_samps = test_x.shape[0]
        interval = self.train_basis_intervals[0][0]
        predicted_mean = np.zeros((n_samps, self.dy))
        if self.adaptive_inputs is True:
            if self.full_x is None:
                if isinstance(self.input_obj.input_model, list) is True:
                    test_x_ = []
                    for _ in range(len(self.input_obj.input_model)):
                        test_x_.append(self.input_obj.input_model[_].predict(test_x))
                    test_x = np.mean(test_x_, axis=0)
                else:
                    test_x = self.input_obj.input_model.predict(test_x)
            else:
                test_x = self.input_obj.z
        x_test = self._get_test_inputs(x=test_x, resolution=0, region=0, index_set=None)
        for i in range(self.n_basis):
            basis_id = i + 1
            phi_x_test, _ = self.basis_function_obj[0].get_eigenpairs(x=x_test,
                                                                      basis_id=basis_id,
                                                                      basis_interval=interval)
            predicted_mean += np.tile(phi_x_test, (self.dy, 1)).T * \
                self.stats_obj[0].scale_axis_mean[0][:, i]
        if self.bias_region_specific is True:
            mu = self.stats_obj[0].bias_mean[0]
        else:
            mu = self.stats_obj[0].bias_mean
        predicted_mean += mu
        return predicted_mean

    def _get_get_predicted_mean(self, test_x, index_set, number_of_regions):
        if index_set.get_n_resolutions() > self.index_set_obj.get_n_resolutions():
            raise ValueError('resolution in the test index set must be smaller or equal to that in the '
                             'train set.')
        if number_of_regions is None:
            if self.index_set_obj.divider != index_set.divider:
                raise ValueError('divider on the training index_set must be'
                                 ' the same as in the test index_set.')
        if number_of_regions is not None:
            if (self.n_regions == number_of_regions) is False:
                raise ValueError('number of regions in the training must be the same as test.')
        predicted_mean = []
        n_layers = index_set.get_n_resolutions()+1
        if self.adaptive_inputs is True:
            if self.full_x is None:
                if isinstance(self.input_obj.input_model, list) is True:
                    test_x_ = []
                    for _ in range(len(self.input_obj.input_model)):
                        test_x_.append(self.input_obj.input_model[_].predict(test_x))
                    test_x = np.mean(test_x_, axis=0)
                else:
                    test_x = self.input_obj.input_model.predict(test_x)
            else:
                test_x = self.input_obj.z

        for j in range(n_layers):
            predicted_mean_j = []
            for l in range(self.n_regions[j]):
                interval = self.train_basis_intervals[j][l]
                x_test = self._get_test_inputs(x=test_x, resolution=j, region=l,
                                               index_set=index_set.index_set)
                predicted_mean_jl = np.zeros((x_test.shape[0], self.dy))
                for i in range(self.n_basis):
                    basis_id = i + 1
                    phi_x_test, _ = self.basis_function_obj[j].get_eigenpairs(x=x_test,
                                                                              basis_id=basis_id,
                                                                              basis_interval=interval)
                    predicted_mean_jl += np.tile(phi_x_test, (self.dy, 1)).T * \
                        self.stats_obj[j].scale_axis_mean[l][:, i]
                if self.bias_region_specific is True:
                    mu = self.stats_obj[j].bias_mean[l]
                else:
                    mu = self.stats_obj[j].bias_mean
                predicted_mean_jl += mu
                predicted_mean_j.append(predicted_mean_jl)
            predicted_mean.append(np.concatenate(predicted_mean_j))
        return sum(predicted_mean)

    def get_predicted_mean(self, test_x, index_set_obj=None, number_of_regions=None):
        if self.standard_normalized_inputs is True:
            test_x = (test_x - np.full(test_x.shape, self.mean_x_train)) / \
            np.full(test_x.shape, self.std_x_train)
        if index_set_obj is None:
            return self._get_get_predicted_mean_global(test_x)
        else:
            return self._get_get_predicted_mean(test_x,
                                                index_set=index_set_obj,
                                                number_of_regions=number_of_regions)

    def get_central_moment2(self, test_x, index_set_obj=None, number_of_regions=None):
        if self.standard_normalized_inputs is True:
            test_x = (test_x - np.full(test_x.shape, self.mean_x_train)) / \
            np.full(test_x.shape, self.std_x_train)
        if index_set_obj is None:
            return self._get_get_predicted_var_global(test_x)
        else:
            return self._get_get_predicted_var(test_x, index_set_obj, number_of_regions)

    def get_test_likelihood(self, test, index_set_obj=None, number_of_regions=None):
        test_x = test[0]
        test_y = test[1]
        mf = self.get_predicted_mean(test_x, index_set_obj, number_of_regions)
        vf = self.get_central_moment2(test_x, index_set_obj=index_set_obj)
        ll = -0.5 * np.log(2 * np.pi * vf) - 0.5 * (np.linalg.norm((test_y - mf), axis=1)**2)/vf
        return np.mean(ll)

    def _get_get_predicted_var_global(self, test_x):
        n_samps = test_x.shape[0]
        interval = self.train_basis_intervals[0][0]
        predicted_central_moment2 = np.zeros(n_samps)
        if self.adaptive_inputs is True:
            if self.full_x is None:
                if isinstance(self.input_obj.input_model, list) is True:
                    test_x_ = []
                    for _ in range(len(self.input_obj.input_model)):
                        test_x_.append(self.input_obj.input_model[_].predict(test_x))
                    test_x = np.mean(test_x_, axis=0)
                else:
                    test_x = self.input_obj.input_model.predict(test_x)
            else:
                test_x = self.input_obj.z
        x_test = self._get_test_inputs(x=test_x, resolution=0, region=0, index_set=None)
        stats = self.stats_obj[0]
        for i in range(self.n_basis):
            basis_id = i + 1
            phi_x_test, _ = self.basis_function_obj[0].get_eigenpairs(x=x_test,
                                                                      basis_id=basis_id,
                                                                      basis_interval=interval)
            predicted_central_moment2 += stats.scale_axis_central_moment2[0][i] * (phi_x_test**2)
        if self.bias_region_specific is True:
            bias_var = stats.bias_var[0]
        else:
            bias_var = stats.bias_var
        predicted_central_moment2 += bias_var
        return predicted_central_moment2

    def _get_get_predicted_var(self, test_x, index_set_obj, number_of_regions):
        if self.adaptive_inputs is True:
            if self.full_x is None:
                if isinstance(self.input_obj.input_model, list) is True:
                    test_x_ = []
                    for _ in range(len(self.input_obj.input_model)):
                        test_x_.append(self.input_obj.input_model[_].predict(test_x))
                    test_x = np.mean(test_x_, axis=0)
                else:
                    test_x = self.input_obj.input_model.predict(test_x)
            else:
                test_x = self.input_obj.z
        phi_x_test = []
        n_samps_test = []
        for j in range(self.n_layers):
            phi_x_test_j = []
            n_samps_j = []
            for l in range(self.n_regions[j]):
                interval = self.train_basis_intervals[j][l]
                x_test = self._get_test_inputs(x=test_x, resolution=j, region=l, index_set=index_set_obj.index_set)
                n_samps_j.append(len(index_set_obj.index_set[j][l]))
                phi_x_test_jl = np.zeros((n_samps_j[l], self.n_basis))
                for i in range(self.n_basis):
                    basis_id = i + 1
                    phi_x_test_jl[:, i], _ = self.basis_function_obj[j].get_eigenpairs(x=x_test,
                                                                                       basis_id=basis_id,
                                                                                       basis_interval=interval)
                phi_x_test_j.append(phi_x_test_jl)
            phi_x_test.append(phi_x_test_j)
            n_samps_test.append(n_samps_j)
        for j in range(self.n_layers):
            self.stats_obj[j].initialize_latent_functions(n_samps_test[j])
        for j in range(self.n_layers):
            next_resolution = j + 1
            if next_resolution < self.n_layers:
                self.stats_obj[next_resolution].update_latent_functions(resolution=next_resolution,
                                                                        index_set=index_set_obj.index_set,
                                                                        stats=self.stats_obj,
                                                                        phi_x=phi_x_test)
        pred_var = []
        for j in range(self.n_layers):
            pred_var.append(self._get_var(j, phi_x_test[j]))
        return sum(pred_var)

    def _get_var(self, res, phi_x_test):
        j = res
        phi_x = phi_x_test
        y_mean, y_var = self.output_obj.infer_stats(stats=self.stats_obj[j], basis_functions=phi_x)
        stats = self.stats_obj[j]
        var_j = []
        for l in range(self.n_regions[j]):
            n_samps_l = phi_x[l].shape[0]
            phi_x_l = phi_x[l]
            y_l = y_mean[l]
            y_var_l = y_var[l] * n_samps_l
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            if self.bias_region_specific is True:
                bias_mean = stats.bias_mean[l]
                bias_var = stats.bias_var[l]
            else:
                bias_mean = stats.bias_mean
                bias_var = stats.bias_var

            mean_term_l = np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l] - bias_mean, axis=1)**2
            var_f_l = stats.latent_f_var[l][0]
            var_au_l = np.sum(phi_x_l**2 * stats.scale_axis_central_moment2[l], axis=1)
            var_j.append(mean_term_l + var_f_l + var_au_l + y_var_l + bias_var)
        return np.concatenate(var_j)

    def _test_data_likelihood(self, res, phi_x_test):
        j = res
        phi_x = phi_x_test
        y_mean, y_var = self.output_obj.infer_stats(stats=self.stats_obj[j], basis_functions=phi_x)
        stats = self.stats_obj[j]
        log_ll_j = 0
        const = 0
        for l in range(self.n_regions[j]):
            n_samps_l = phi_x[l].shape[0]
            phi_x_l = phi_x[l]
            y_l = y_mean[l]
            y_var_l = y_var[l] * n_samps_l
            sum_term1 = np.zeros_like(y_l)
            for i in range(self.n_basis):
                sum_term1 += stats.scale_axis_mean[l][:, i] * np.tile(phi_x_l[:, i], (self.dy, 1)).T
            mean_term_l = np.sum(np.linalg.norm(y_l - sum_term1 - stats.latent_f_mean[l], axis=1)**2)
            var_f_l = np.sum(stats.latent_f_var[l])
            var_au_l = np.sum((phi_x_l**2) * stats.scale_axis_central_moment2[l])
            log_ll_j += mean_term_l + var_f_l + var_au_l + y_var_l
            if self.noise_region_specific is True:
                noise_log_mean = stats.noise_log_mean[l]
            else:
                noise_log_mean = stats.noise_log_mean
            const += 0.5*self.dy * (noise_log_mean - np.log(2*np.pi)) * n_samps_l
        log_ll_j += const
        return log_ll_j

    def _get_test_inputs(self, x, resolution, region, index_set):
        if index_set is not None:
            x = x[index_set[resolution][region], :]
        return x

    @staticmethod
    def _compute_initial_noise_var_from_snr(y, snr_ratio):
        n_samps = y.shape[0]
        y_var = (np.linalg.norm(y)**2)/n_samps - np.dot(np.mean(y, axis=0), np.mean(y, axis=0))
        noise_var = y_var/snr_ratio
        return noise_var

    def get_basis_contributions(self):
        basis_contributions = []
        for j in range(self.n_layers):
            n_regions = self.n_regions[j]
            basis_contr_region = []
            for l in range(n_regions):
                basis_contr_region.append(
                    self.stats_obj[j].scale_moment2[l]/sum(self.stats_obj[j].scale_moment2[l]))
            basis_contributions.append(basis_contr_region)
        return basis_contributions
