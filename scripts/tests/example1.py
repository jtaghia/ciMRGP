import numpy as np

from IndexSetGenerator import IndexSetUniform
from KernelClass import LaplacianEigenpairs, MaternKernel
from MRGP import MultiResolutionGaussianProcess
from BasisInterval import BasisInterval
from PlotClass import RegressionPlot
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
np.random.seed(210776)


def f(x):
    """The function to predict. One dimensional input and two dimensional output"""
    y1 = np.log(np.log(x)+abs(np.sin(x**2)*np.exp(np.sin(np.cos(2*x)))))
    y2 = np.log(np.log(x)+abs(np.sin(-x**2 + 3*x+5) + np.log(1+abs(np.cos(x**2)))))
    return np.array([y1, y2])


def mll(y_test, mean_pred, var_pred):
    mf = mean_pred
    vf = var_pred
    ll = -0.5 * np.log(2 * np.pi * vf) - 0.5 * (np.linalg.norm((y_test - mf), axis=1)**2)/vf
    return np.mean(ll)


X = np.atleast_2d(np.linspace(1, 3, 1*32)).T
y = f(x=X)
y = y[:, :, 0].T
dy = 1 + 2 * np.random.random(y.shape)
noise = .1*np.random.normal(0, dy)
y += noise
train = [X, y]
x = np.atleast_2d(np.linspace(1, 3, 100000)).T
test = [x, f(x=x)[:, :, 0].T]

# full_x = test[0]
full_x = None

pl = RegressionPlot()

# learning
n_res = 5
divider = 2
basis_function = LaplacianEigenpairs()
prior_influence = 1e0
spectral_density = MaternKernel(nu=.1, l=1, sf=prior_influence)
index_set = IndexSetUniform(sample_length=train[0].shape[0], resolution=n_res, divider=divider)
basis_interval = BasisInterval(opt_interval_factor=(1, 1.1))
# basis_interval = None
mrgp = MultiResolutionGaussianProcess(train_xy=train,
                                      n_basis=20,
                                      index_set_obj=index_set,
                                      basis_function_obj=basis_function,
                                      spectral_density_obj=spectral_density,
                                      adaptive_inputs=True,
                                      standard_normalized_inputs=True,
                                      basis_interval_obj=basis_interval,
                                      interval_factor=1,
                                      axis_resolution_specific=False,
                                      ard_resolution_specific=False,
                                      noise_region_specific=True,
                                      bias_region_specific=True,
                                      noninformative_initialization=True,
                                      snr_ratio=None,
                                      full_x=None,
                                      forced_independence=False,
                                      verbose=True)

mrgp.fit(30, None)
index_set_test = IndexSetUniform(sample_length=test[0].shape[0], resolution=n_res, divider=divider)
preds_mean = mrgp.get_predicted_mean(test_x=test[0],
                                     index_set_obj=index_set_test)
# print("test log-likelihood %s:" % str(mrgp.get_test_likelihood(test=test, index_set_obj=None)))
preds_var1 = mrgp.get_central_moment2(test_x=test[0],
                                      index_set_obj=index_set_test,
                                      number_of_regions=None)

preds_var = mrgp.get_central_moment2(test_x=test[0],
                                      index_set_obj=None,
                                      number_of_regions=None)
mll = mll(y_test=test[0], mean_pred=preds_mean, var_pred=preds_var1)
print("mean log likelihood 2 %s" % str(mll))


pl.plot_data(train=train, test=test, predictions=preds_mean, fig_name='train_test')


print("R2-score %s:" % str(r2_score(y_pred=preds_mean, y_true=test[1])))
print("RMSE %s:" % str(mean_squared_error(y_pred=preds_mean, y_true=test[1])))


plt.show()
