import numpy as np
from CommonDensities import Bingham


EPSILON = 1e-45


class SharedPrior(object):
    def __init__(self, n_basis, dy):
        self.axis_bingham_b = np.zeros((n_basis, dy, dy))
        self.axis_bingham_kappa = np.zeros((n_basis, dy))
        self.axis_bingham_rho = np.zeros((n_basis, dy))
        self.axis_bingham_axes = np.zeros((n_basis, dy, dy))
        self.axis_bingham_log_const = np.zeros(n_basis)

        self.ard_gamma_shape = np.zeros(n_basis)
        self.ard_gamma_scale = np.zeros(n_basis)

        self.n_basis = n_basis
        self.dy = dy

    #  BASIS AXIS
    def basis_axis(self, noninformative=True):
        if noninformative is True:
            self._axis_basis_noninformative()
        else:
            raise ValueError('not yet implemented...')

    def _axis_basis_noninformative(self):
        for i in range(self.n_basis):
            b = np.zeros((self.dy, self.dy))
            bingham_pd = Bingham(b)
            self.axis_bingham_kappa[i, :] = bingham_pd.kappa
            self.axis_bingham_rho[i, :] = bingham_pd.rho
            self.axis_bingham_axes[i, :, :] = bingham_pd.axes
            self.axis_bingham_log_const[i] = bingham_pd.log_const

    #  ARD
    def ard(self, prior_influence=1., noninformative=True):
        #  TODO: for later versions, "prior_influence" should be removed...
        """
        :param noninformative: set to True for noninformative initialization
        :param prior_influence: prior_influence-->0 pushes the prior on scales towards
        a non-regularized basis function (default: 1.0)
        """
        if noninformative is True:
            self._ard_noninformative(prior_influence)
        else:
            raise ValueError('not yet implemented...')

    def _ard_noninformative(self, prior_influence):
        self.ard_gamma_shape = EPSILON*np.ones(self.n_basis)
        self.ard_gamma_scale = self.ard_gamma_shape/prior_influence


class Prior(object):
    def __init__(self, n_basis, dy, n_regions):

        self.scale_precision = None

        self.noise_region_specific = None
        self.noise_gamma_scale = None
        self.noise_gamma_shape = None

        self.bias_region_specific = None
        self.bias_normal_mean = None
        self.bias_normal_precision = None

        self.n_basis = n_basis
        self.dy = dy
        self.n_regions = n_regions

    # BASIS AXIS SCALE
    def basis_axis_scale(self, spectral_density, noninformative=True):
        if noninformative is True:
            self.scale_precision = []
            for l in range(self.n_regions):
                self.scale_precision.append(1/spectral_density[l])
        else:
            raise ValueError('not yet implemented...')

    #  NOISE
    def noise(self, noise_var=1., region_specific=True, noninformative=True):
        self.noise_region_specific = region_specific
        if region_specific is False:
            if noninformative is True:
                self.noise_gamma_scale, self.noise_gamma_shape = self._noise_noninformative(noise_var)
            else:
                raise ValueError('not yet implemented...')
        elif region_specific is True:
            if noninformative is True:
                self.noise_gamma_scale = []
                self.noise_gamma_shape = []
                for l in range(self.n_regions):
                    noise_gamma_scale, noise_gamma_shape = self._noise_noninformative(noise_var)
                    self.noise_gamma_scale.append(noise_gamma_scale)
                    self.noise_gamma_shape.append(noise_gamma_shape)
            else:
                raise ValueError('not yet implemented...')
        else:
            raise TypeError('region_specific can be either True or False.')

    @staticmethod
    def _noise_noninformative(noise_var):
        noise_gamma_shape = EPSILON
        if noise_var is None:
            noise_var = 1.
        noise_gamma_scale = (noise_gamma_shape+1) * noise_var
        return noise_gamma_scale, noise_gamma_shape

    #  BIAS
    def bias(self, region_specific=True, noninformative=True):
        self.bias_region_specific = region_specific
        if region_specific is False:
            if noninformative is True:
                self.bias_normal_mean, self.bias_normal_precision = self._bias_noninformative()
            else:
                raise ValueError('not yet implemented...')
        elif region_specific is True:
            if noninformative is True:
                self.bias_normal_mean = []
                self.bias_normal_precision = []
                for l in range(self.n_regions):
                    bias_normal_mean, bias_normal_precision = self._bias_noninformative()
                    self.bias_normal_mean.append(bias_normal_mean)
                    self.bias_normal_precision.append(bias_normal_precision)
            else:
                raise ValueError('not yet implemented...')
        else:
            raise TypeError('region_specific can be either True or False.')

    def _bias_noninformative(self):
        bias_normal_mean = np.zeros(self.dy)
        bias_normal_precision = EPSILON
        return bias_normal_mean, bias_normal_precision


class IndependentPrior(object):
    def __init__(self, n_basis, dy, n_regions):

        self.axis_bingham_b = None
        self.axis_bingham_kappa = None
        self.axis_bingham_rho = None
        self.axis_bingham_axes = None
        self.axis_bingham_log_const = None

        self.ard_gamma_shape = None
        self.ard_gamma_scale = None

        self.scale_precision = None

        self.noise_region_specific = None
        self.noise_gamma_scale = None
        self.noise_gamma_shape = None

        self.bias_region_specific = None
        self.bias_normal_mean = None
        self.bias_normal_precision = None

        self.n_basis = n_basis
        self.dy = dy
        self.n_regions = n_regions


    #  BASIS AXIS
    def basis_axis(self, noninformative=True):
        if noninformative is True:
            self._axis_basis_noninformative()
        else:
            raise ValueError('not yet implemented...')

    def _axis_basis_noninformative(self):
        self.axis_bingham_b = []
        self.axis_bingham_kappa = []
        self.axis_bingham_rho = []
        self.axis_bingham_axes = []
        self.axis_bingham_log_const = []
        for l in range(self.n_regions):
            self.axis_bingham_b.append(np.zeros((self.n_basis, self.dy, self.dy)))
            axis_bingham_kappa = np.zeros((self.n_basis, self.dy))
            axis_bingham_rho = np.zeros((self.n_basis, self.dy))
            axis_bingham_axes = np.zeros((self.n_basis, self.dy, self.dy))
            axis_bingham_log_const = np.zeros(self.n_basis)
            for i in range(self.n_basis):
                b = np.zeros((self.dy, self.dy))
                bingham_pd = Bingham(b)
                axis_bingham_kappa[i, :] = bingham_pd.kappa
                axis_bingham_rho[i, :] = bingham_pd.rho
                axis_bingham_axes[i, :, :] = bingham_pd.axes
                axis_bingham_log_const[i] = bingham_pd.log_const
            self.axis_bingham_kappa.append(axis_bingham_kappa)
            self.axis_bingham_rho.append(axis_bingham_rho)
            self.axis_bingham_axes.append(axis_bingham_axes)
            self.axis_bingham_log_const.append(axis_bingham_log_const)

    #  ARD
    def ard(self, prior_influence=1., noninformative=True):
        #  TODO: for later versions, "prior_influence" should be removed...
        """
        :param noninformative: set to True for noninformative initialization
        :param prior_influence: prior_influence-->0 pushes the prior on scales towards
        a non-regularized basis function (default: 1.0)
        """
        if noninformative is True:
            self._ard_noninformative(prior_influence)
        else:
            raise ValueError('not yet implemented...')

    def _ard_noninformative(self, prior_influence):
        self.ard_gamma_shape = []
        self.ard_gamma_scale = []
        for l in range(self.n_regions):
            self.ard_gamma_shape.append(EPSILON*np.ones(self.n_basis))
            self.ard_gamma_scale.append(self.ard_gamma_shape[l]/prior_influence)

    # BASIS AXIS SCALE
    def basis_axis_scale(self, spectral_density, noninformative=True):
        if noninformative is True:
            self.scale_precision = []
            for l in range(self.n_regions):
                self.scale_precision.append(1 / spectral_density[l])
        else:
            raise ValueError('not yet implemented...')

    #  NOISE
    def noise(self, noise_var=1., region_specific=True, noninformative=True):
        self.noise_region_specific = region_specific
        if region_specific is False:
            if noninformative is True:
                self.noise_gamma_scale, self.noise_gamma_shape = self._noise_noninformative(noise_var)
            else:
                raise ValueError('not yet implemented...')
        elif region_specific is True:
            if noninformative is True:
                self.noise_gamma_scale = []
                self.noise_gamma_shape = []
                for l in range(self.n_regions):
                    noise_gamma_scale, noise_gamma_shape = self._noise_noninformative(noise_var)
                    self.noise_gamma_scale.append(noise_gamma_scale)
                    self.noise_gamma_shape.append(noise_gamma_shape)
            else:
                raise ValueError('not yet implemented...')
        else:
            raise TypeError('region_specific can be either True or False.')

    @staticmethod
    def _noise_noninformative(noise_var):
        noise_gamma_shape = EPSILON
        if noise_var is None:
            noise_var = 1.
        noise_gamma_scale = (noise_gamma_shape + 1) * noise_var
        return noise_gamma_scale, noise_gamma_shape

    #  BIAS
    def bias(self, region_specific=True, noninformative=True):
        self.bias_region_specific = region_specific
        if region_specific is False:
            if noninformative is True:
                self.bias_normal_mean, self.bias_normal_precision = self._bias_noninformative()
            else:
                raise ValueError('not yet implemented...')
        elif region_specific is True:
            if noninformative is True:
                self.bias_normal_mean = []
                self.bias_normal_precision = []
                for l in range(self.n_regions):
                    bias_normal_mean, bias_normal_precision = self._bias_noninformative()
                    self.bias_normal_mean.append(bias_normal_mean)
                    self.bias_normal_precision.append(bias_normal_precision)
            else:
                raise ValueError('not yet implemented...')
        else:
            raise TypeError('region_specific can be either True or False.')

    def _bias_noninformative(self):
        bias_normal_mean = np.zeros(self.dy)
        bias_normal_precision = EPSILON
        return bias_normal_mean, bias_normal_precision

