import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from IndexSetGenerator import IndexSetUniform
from KernelClass import LaplacianEigenpairs, MaternKernel
from MRGP import MultiResolutionGaussianProcess
from BasisInterval import BasisInterval


def f(x):
    """The function to predict. One dimensional input and two dimensional output"""
    y1 = np.log(np.log(x)+abs(np.sin(x**2)*np.exp(np.sin(np.cos(2*x)))))
    y2 = np.log(np.log(x)+abs(np.sin(-x**2 + 3*x+5) + np.log(1+abs(np.cos(x**2)))))
    return np.array([y1, y2])


def generate_data():
    X = np.atleast_2d(np.linspace(1, 3, 1*32)).T
    y = f(x=X)
    y = y[:, :, 0].T
    dy = 1 + 2 * np.random.random(y.shape)
    noise = .1*np.random.normal(0, dy)
    y += noise
    train = [X, y]
    x = np.atleast_2d(np.linspace(1, 3, 100000)).T
    test = [x, f(x=x)[:, :, 0].T]
    return train, test


class MRGP():
    def __init__(self, n_res, divider, n_basis, n_iter, forced_independence):
        if forced_independence is True:
            model_name = 'fiMRGP'
        else:
            model_name = 'ciMRGP'

        basis_function = LaplacianEigenpairs()
        spectral_density = MaternKernel(nu=.1, l=1, sf=1)
        index_set = IndexSetUniform(sample_length=train[0].shape[0], resolution=n_res, divider=divider)
        basis_interval = BasisInterval(opt_interval_factor=(1, 1.2))
        mrgp = MultiResolutionGaussianProcess(train_xy=train,
                                              n_basis=n_basis,
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
                                              forced_independence=forced_independence,
                                              snr_ratio=None,
                                              verbose=True)

        mrgp.fit(n_iter, None)
        index_set_test = IndexSetUniform(sample_length=test[0].shape[0], resolution=n_res, divider=divider)
        preds_mean = mrgp.get_predicted_mean(test_x=test[0], index_set_obj=index_set_test)

        preds_var = mrgp.get_central_moment2(test_x=test[0], index_set_obj=None, number_of_regions=None)

        mll = self._return_mll(y_test=test[0], mean_pred=preds_mean, var_pred=preds_var)
        print(model_name + ": average log likelihood%s" % str(mll))
        print(model_name + ": R2-score %s:" % str(r2_score(y_pred=preds_mean, y_true=test[1])))
        print(model_name + ": RMSE %s:" % str(mean_squared_error(y_pred=preds_mean, y_true=test[1])))

        #  ----------------------------
        colors = sns.color_palette("Set2", 10)

        plt.figure("1dx_2dy")
        dy = train[1].shape[1]
        for d in range(dy):
            plt.plot(test[0], test[1][:, d],
                     color=colors[d], linestyle=':',
                     label=r'$f_' + str(d + 1) + '(x)$', linewidth=1, alpha=1)
            plt.plot(train[0],
                     train[1][:, d],
                     color=colors[d], marker='o', linestyle='none', markersize=4, alpha=0.5,
                     label=r'${y_' + str(d + 1) + '}$')
            plt.plot(test[0],
                     preds_mean[:, d],
                     color=colors[d], linewidth=1, label=r'$\hat{f}_' + str(d + 1) + '(x)$')
            plt.ylim([1.5 * np.min(test[1]), 1.5 * np.max(test[1])])
            plt.legend(loc='upper left')
            plt.xticks([])
            plt.yticks([])

        plt.savefig('./figs/ciMRGP_vs_fiMRGP/' + model_name + str(n_res))
        plt.close()

    def _return_mll(self, y_test, mean_pred, var_pred):
        mf = mean_pred
        vf = var_pred
        ll = -0.5 * np.log(2 * np.pi * vf) - 0.5 * (np.linalg.norm((y_test - mf), axis=1)**2)/vf
        return np.mean(ll)


# generate data
train, test = generate_data()

# model specifications
n_res = 5
divider = 2
n_basis = 20
n_iter = 15

#  conditionally independent MRGP (ciMRGP)
ciMRGP = MRGP(n_res, divider, n_basis, n_iter, forced_independence=False)

#  fully independent MRGP (fiMRGP)
fiMRGP = MRGP(n_res, divider, n_basis, n_iter, forced_independence=True)