"""This module contains the functions useful for the generation of figures."""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np


def create_surface_plot(fun, fname='fig-surface.png'):
    """This function creates a beautiful surface plot."""
    ax = plt.axes(projection='3d')

    v_1 = np.linspace(0.0, 1.0, 200, endpoint=True)
    v_2 = np.linspace(0.0, 1.0, 200, endpoint=True)
    v_1, v_2 = np.meshgrid(v_1, v_2)

    levels = np.linspace(0.0, 1.0, 11, endpoint=True)
    cmap, norm = _create_discrete_color_jet(10)
    cp = ax.plot_surface(v_1, v_2, fun(v_1, v_2, True), rstride=1, cstride=1, cmap=cmap, norm=norm)
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.zaxis.get_major_ticks()[0].set_visible(False)

    plt.colorbar(cp, ticks=levels)

    ax.set_zlim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'Attribute $x$')
    ax.set_ylabel(r'Attribute $y$')
    plt.savefig(fname)


def create_contour_plot(fun, fname='fig-contour.png'):
    """This function creates a beautiful contour plot."""
    ax = plt.figure().add_subplot(111)

    v_1 = np.linspace(0.0, 1.0, 1000, endpoint=True)
    v_2 = np.linspace(0.0, 1.0, 1000, endpoint=True)
    v_1, v_2 = np.meshgrid(v_1, v_2)

    levels = np.linspace(0.0, 1.0, 11, endpoint=True)
    cmap, norm = _create_discrete_color_jet(10)
    cp = plt.contourf(v_1, v_2, fun(v_1, v_2, True), levels=levels, cmap=cmap, norm=norm)
    ax.yaxis.get_major_ticks()[0].set_visible(False)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'Attribute $x$')
    ax.set_ylabel(r'Attribute $y$')

    plt.colorbar(cp, ticks=levels)

    plt.savefig(fname)


def _create_discrete_color_jet(num_colors):
    """This function ensures discrete contours."""
    cmap = cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom jet', cmaplist, cmap.N)
    bounds = np.linspace(0, 1, num_colors + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm
