"""This module contains the functions useful for the generation of figures."""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np

from copulpy.clsUtilityCopula import UtilityCopulaCls


def create_uniattribute_plot(copula_spec, which, fname='fig-uniattribute'):
    """This function creates a line plot of the univariate utility function."""
    grid = np.tile(0.0, (1000, 2))

    if which in ['x']:
        upper, col = copula_spec['bounds'][0], 0
    elif which in ['y']:
        upper, col = copula_spec['bounds'][1], 1
    else:
        raise NotImplementedError
    x_axis_values = np.linspace(0.01, upper, 1000, endpoint=True)
    grid[:, col] = x_axis_values

    copula = UtilityCopulaCls(copula_spec)
    y_values = np.tile(np.nan, 1000)
    for i in range(1000):
        y_values[i] = copula.evaluate(*grid[i, :])

    ax = plt.figure().add_subplot(111)
    ax.plot(x_axis_values, y_values)
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'Attribute $' + which + '$')
    ax.set_ylabel(r'Utility')
    plt.savefig(fname + '.png')


def create_surface_plot(copula_spec, is_normalized, fname='fig-surface'):
    """This function creates a beautiful surface plot."""
    fun = UtilityCopulaCls(copula_spec).evaluate

    if is_normalized:
        x_upper, y_upper = 1, 1
    else:
        x_upper, y_upper = copula_spec['bounds']

    v = []
    for upper in [x_upper, y_upper]:
        v += [np.linspace(0.01, upper, 200, endpoint=True)]
    v = np.meshgrid(*v)
    plt.figure()
    ax = plt.axes(projection='3d')

    levels = np.linspace(0.01, 1.0, 11, endpoint=True)
    cmap, norm = _create_discrete_color_jet(10)
    cp = ax.plot_surface(*v, fun(*v, is_normalized), rstride=1, cstride=1, cmap=cmap, norm=norm)
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.zaxis.get_major_ticks()[0].set_visible(False)

    plt.colorbar(cp, ticks=levels)

    ax.set_zlim(0, 1)
    ax.set_xlim(0, x_upper)
    ax.set_ylim(0, y_upper)
    ax.set_xlabel(r'Attribute $x$')
    ax.set_ylabel(r'Attribute $y$')
    plt.savefig(fname + '.png')


def create_contour_plot(copula_spec, is_normalized, fname='fig-contour'):
    """This function creates a beautiful contour plot."""
    fun = UtilityCopulaCls(copula_spec).evaluate

    if is_normalized:
        x_upper, y_upper = 1, 1
    else:
        x_upper, y_upper = copula_spec['bounds']

    v = []
    for upper in [x_upper, y_upper]:
        v += [np.linspace(0.0, upper, 1000, endpoint=True)]
    v = np.meshgrid(*v)

    ax = plt.figure().add_subplot(111)

    levels = np.linspace(0.0, 1.0, 11, endpoint=True)
    cmap, norm = _create_discrete_color_jet(10)
    cp = plt.contourf(*v, fun(*v, is_normalized), levels=levels, cmap=cmap, norm=norm)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    ax.set_xlim(0, x_upper)
    ax.set_ylim(0, y_upper)

    ax.set_xlabel(r'Attribute $x$')
    ax.set_ylabel(r'Attribute $y$')

    plt.colorbar(cp, ticks=levels)

    plt.savefig(fname + '.png')


def _create_discrete_color_jet(num_colors):
    """This function ensures discrete contours."""
    cmap = cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom jet', cmaplist, cmap.N)
    bounds = np.linspace(0, 1, num_colors + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm
