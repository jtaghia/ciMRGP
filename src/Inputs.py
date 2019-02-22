from RegressionInput import *
import random
from tqdm import *
from IndexSetGenerator import IndexSetUniform


class Inputs(object):
    def __init__(self, x, index_set, learn_inputs, full_x, input_model):
        self.learn_inputs = learn_inputs
        self.index_set = index_set
        if self.learn_inputs is True:
            z = np.atleast_2d(np.linspace(start=np.min(x), stop=np.max(x), num=x.shape[0])).T
            if full_x is None:
                train_data = [x, z]
            else:
                z_full = np.atleast_2d(np.linspace(start=np.min(full_x), stop=np.max(full_x), num=full_x.shape[0])).T
                train_data = [full_x, z_full]

            if input_model is None:
                if train_data[0].shape[0] < 3001:
                    self.input_model = GP_RBF()
                    self.input_model.fit(train_data)
                else:
                    n_samps = train_data[0].shape[0]
                    n_repeats = 1
                    min_length = 3000
                    n_divide = self._get_best_divider(n_samps, rate=random.uniform(.1, .2))
                    index_set_obj = IndexSetUniform(sample_length=n_samps, resolution=1, divider=n_divide)
                    index_set = index_set_obj.index_set[-1]
                    n_regions = len(index_set)
                    ids = []
                    ids_all = list(range(n_samps))
                    for rep in range(n_repeats):
                        ids_l = []
                        for l in range(n_regions):
                            ids_l.append(np.random.permutation(index_set[l])[0])
                        rem_ids = np.delete(ids_all, ids_l)
                        if min_length > len(ids_l):
                            rand_ids = np.random.permutation(rem_ids)[0:min_length - len(ids_l)]
                            ids_rep = list(rand_ids) + ids_l
                        else:
                            ids_rep = ids_l
                        ids.append(list(np.sort(np.unique(ids_rep))))
                    self.input_model = []
                    for _ in tqdm(range(len(ids)), desc='initial learning'):
                        train_data_ = [train_data[0][ids[_], :], train_data[1][ids[_], :]]
                        self.input_model.append(GP_RBF())
                        self.input_model[_].fit(train_data_)
            else:
                self.input_model = input_model
            self.z = train_data[1]
            self.x = z

        else:
            self.x = x

    def get_inputs(self, resolution, region):
        T_jl = self.index_set.index_set[resolution][region]
        x = self.x[T_jl, :]
        return x

    def _get_best_divider(self, n_samps, rate=0.2):
        if n_samps < 10000:
            factor = 1 * rate
        elif n_samps < 100000:
            factor = 1e-1 * rate
        elif n_samps < 1000000:
            factor = 1e-2 * rate
        else:
            factor = 1e-3 * rate
        return int(np.floor(factor*n_samps))
