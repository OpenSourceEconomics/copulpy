#!/usr/bin/env python
"""This module creates a set of figures"""
from auxiliary_plots import create_uniattribute_plot
from auxiliary_plots import create_surface_plot
from auxiliary_plots import create_contour_plot
from copulpy.clsPower import PowerCls

if __name__ == '__main__':

    copula_spec = dict()

    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'

    copula_spec['bounds'] = [150, 150]
    copula_spec['u'] = [1.0, 0.0]
    copula_spec['delta'] = 1
    copula_spec['r'] = 0.5
    copula_spec['a'] = 1
    copula_spec['b'] = 0

    bounds = copula_spec['bounds']
    r = copula_spec['r']
    a = copula_spec['a']
    b = copula_spec['b']

    copula_spec['x_uniattribute_utility'] = PowerCls(r, a, b, bounds[0])
    copula_spec['y_uniattribute_utility'] = PowerCls(r, a, b, bounds[1])

    for is_normalized in [True, False]:
        fname = 'fig-contour'
        if is_normalized:
            fname += '-normalized'
        create_contour_plot(copula_spec, is_normalized, fname)

    for is_normalized in [True, False]:
        fname = 'fig-surface'
        if is_normalized:
            fname += '-normalized'
        create_surface_plot(copula_spec, is_normalized, fname)

    for which in ['x', 'y']:
        create_uniattribute_plot(copula_spec, which)
