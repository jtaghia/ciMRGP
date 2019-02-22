import abc
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D


class PlotCalss(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, color_set=None, font_scale=None, style=None, three_dimensional=False):
        self.d3 = three_dimensional
        if color_set is None:
            # self.colors = sns.color_palette("Set2", 1000)
            colors = ["windows blue", "faded green","amber", "greyish", "dusty purple"]
            self.colors = sns.xkcd_palette(colors)
        else:
            self.colors = color_set
        if font_scale is None:
            sns.set(font_scale=.7)
        else:
            sns.set(font_scale=font_scale)
        if style is None:
            sns.set_style("white", {'axes.grid': False})

    @abc.abstractmethod
    def plot_data(self, train, test, predictions, fig_name, figsize=(3, 2)):
        return True


class RegressionPlot(PlotCalss):
    name = '2D'

    def plot_data(self, train, test, predictions, fig_name, figsize=(2, 1.5)):
        plt.figure(fig_name, figsize=figsize)
        ax = plt.subplot(111)
        dy = train[1].shape[1]
        for d in range(dy):
            plt.plot(test[0], test[1][:, d],
                     color=self.colors[d], linestyle=':',
                     label='f' + str(d+1) + '(x)', linewidth=.7, alpha=1)
            plt.plot(train[0],
                     train[1][:, d],
                     color=self.colors[d], marker='o', linestyle='none', markersize=2, alpha=0.3,
                     label='y' + str(d+1))
            plt.plot(test[0],
                     predictions[:, d],
                     color=self.colors[d], linewidth=.7, label='y_pred' + str(d+1))
            plt.ylim([1.8*np.min(test[1]), 1.4*np.max(test[1])])
            # plt.legend(loc='upper left')
            plt.xticks([])
            plt.yticks([])

    def plot_predictions(self, inputs, predictions, fig_name):
        plt.figure(fig_name)
        plt.title(fig_name)
        ax = plt.subplot(111)
        dy = predictions.shape[1]
        for d in range(dy):
            plt.plot(inputs,
                     predictions[:, d],
                     color=self.colors[d], linewidth=1, label='y_pred' + str(d+1))
            # plt.legend(loc='upper left')


class RegressionPlot3D(PlotCalss):
    name = '3D'

    def plot_data(self, train, test, predictions, fig_name, figsize=(3, 2)):
        fig_3d_ = plt.figure(fig_name, figsize=figsize)
        ax = fig_3d_.add_subplot(1, 1, 1, projection='3d')
        dy = train[1].shape[1]
        for d in range(dy):
            ax.scatter(train[0][:, 0], train[0][:, 1], (train[1])[:, d], c=self.colors[d], s=.1, marker='o',
                   label=u'$y_' + str(d+1) + '$', alpha=.3)
            ax.plot_wireframe(test[0][:, 0], test[0][:, 1], (test[1])[:, d],
                  rstride=5, cstride=5, linestyle=':', linewidth=.5, alpha=1,
                  color=self.colors[d], label=u'$f_' + str(d+1) + '$' + u'($x_1, x_2$)')
            ax.plot_wireframe(test[0][:, 0], test[0][:, 1], predictions[:, d],
              rstride=5, cstride=5, linewidth=.5,
              color=self.colors[d])
            # plt.xlabel(u'$x_1$')
            # plt.ylabel(u'$x_2$')
            plt.xticks([])
            plt.yticks([])
            # plt.xticks([np.min(test[0][:, 0]), np.max(test[0][:, 0])])
            # plt.yticks([np.min(test[0][:, 1]), np.max(test[0][:, 1])])
            plt.ylim([1.2*np.min(test[1]), 1.2*np.max(test[1])])
            ax.zaxis.set_visible(False)
            ax.set_zticks([])
            # plt.legend(loc='upper left')
        fig_3d_.tight_layout()
        # plt.title(fig_name)

    def plot_predictions_3d(self, inputs, predictions, fig_name):
        fig_3d_ = plt.figure(fig_name)
        # plt.title(fig_name)
        dy = predictions.shape[1]
        for d in range(dy):
            ax = fig_3d_.add_subplot(1, 1, 1, projection='3d')
            ax.plot_wireframe(inputs[:, 0], inputs[:, 1], predictions[:, d],
                  rstride=10, cstride=10, linewidth=1,
                  color=self.colors[d])