import numpy as np


class IndexSetUniform(object):
    def __init__(self, sample_length, resolution, divider, n_regions=None,
                 min_percentage_of_samples_per_region=None):
        """
        :param sample_length: number of samples
        :param resolution: overall resolution
        :param divider: uniform divider rate at each resolution
        :param n_regions is a list containing number of regions at each resolution. Regions are then
        selected randomly. If given, it will overwrite the divider.
        :param min_percentage_of_samples_per_region: default 25%.
        :return: index sets for given resolution
        """
        self.resolution = int(resolution)
        if n_regions is None:
            if self.resolution == 0:
                self.divider = 0
            else:
                self.divider = int(divider)

        self.sample_length = int(sample_length)
        self.index_set = []
        if n_regions is None:
            for m in range(self.resolution+1):
                self.index_set.append(self._get_index_set(resolution=m))
        else:
            self.region_ind = []
            self.min_number_of_samples_per_region = []
            for m in range(self.resolution+1):
                n_regions_res = n_regions[m]
                if min_percentage_of_samples_per_region is None:
                    percentage = 0.25
                else:
                    percentage = min_percentage_of_samples_per_region
                self.min_number_of_samples_per_region.append(int(np.floor(np.divide(self.sample_length,
                                                             n_regions_res)*percentage)))

                index_set, region_ind = self._get_index_set_random(resolution=m,
                                                                   number_of_regions=n_regions_res)
                self.index_set.append(index_set)
                self.region_ind.append(region_ind)

    def get_n_resolutions(self):
        return self.resolution

    def get_index_set(self, resolution):
        return self.index_set[int(resolution)]

    def _get_index_set(self, resolution):
        divider = self.divider
        sample_length = self.sample_length
        number_of_regions = np.power(divider, resolution)
        samples_per_region = np.floor_divide(sample_length, number_of_regions)
        if samples_per_region < 1:
            raise ValueError('*** Chosen resolution is too large! ***')
        left_out_samples = sample_length - (number_of_regions*samples_per_region)
        index_set = []
        start = 0
        for l in range(number_of_regions-1):
            index_set.append(list(range(start, start+samples_per_region)))
            start += samples_per_region
        index_set.append(list(range(start, start+samples_per_region+left_out_samples)))
        return index_set

    def _get_index_set_random(self, resolution, number_of_regions):
        sample_length = self.sample_length
        index_set = []
        if number_of_regions == 1:
            index_set.append(list(range(sample_length)))
            region_ind = None
        else:
            repeat_flg = 1
            while repeat_flg != 0:
                region_ind = [0]
                for l in range(number_of_regions-1):
                    region_ind.append(np.random.randint(1, sample_length))
                region_ind.append(sample_length)
                region_ind = np.sort(region_ind)
                diff_ = np.diff(region_ind)
                repeat_flg_ = []
                for _ in range(len(diff_)):
                    if diff_[_] < self.min_number_of_samples_per_region[resolution]:
                        repeat_flg_.append(True)
                    else:
                        repeat_flg_.append(False)
                repeat_flg = sum(repeat_flg_)
            index_set = []
            for l in range(number_of_regions):
                index_set.append(list(range(region_ind[l], region_ind[l+1])))
        return index_set, region_ind


