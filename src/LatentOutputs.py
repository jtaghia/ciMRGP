import numpy as np


class LatentOutputs(object):

    def get_stats(self, observations):
        y_mean = observations
        y_var = 0.
        return [y_mean], [y_var]

    def infer_point_stats(self, observations, index_set):
        y_mean = []
        y_var = []
        n_regions = len(index_set)
        for l in range(n_regions):
            y_mean.append(observations[index_set[l], :])
            y_var.append(0.)
        return y_mean, y_var

    def infer_stats(self, stats, basis_functions):
        y_mean = self._infer_mean(stats, basis_functions)
        y_var = self._infer_variance(stats)
        return y_mean, y_var

    def _infer_mean(self, stats, basis_functions):
        E_y = []
        for l in range(stats.n_regions):
            phi_x = basis_functions[l]
            n_samps = phi_x.shape[0]
            E_y_l = np.zeros((n_samps, stats.dy))
            for i in range(stats.n_basis):
                E_y_l += np.tile(phi_x[:, i], (stats.dy, 1)).T * stats.scale_axis_mean[l][:, i]
            if stats.bias_region_specific is True:
                mu = stats.bias_mean[l]
            else:
                mu = stats.bias_mean
            E_y_l += (mu + stats.latent_f_mean[l])
            E_y.append(E_y_l)
        return E_y

    def _infer_variance(self, stats):
        y_var = []
        for l in range(stats.n_regions):
            if stats.noise_region_specific is True:
                noise_mean = stats.noise_mean[l]
            else:
                noise_mean = stats.noise_mean
            y_var.append(1./noise_mean)
        return y_var

