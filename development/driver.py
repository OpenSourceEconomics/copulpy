#!/usr/bin/env python
"""This module allows to explore the issues around using multiattribute utility copulas."""
import numpy as np

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.clsPower import PowerCls


def expected_utility_a(r, delta, u_1, u_2, lottery):

    copula_spec = dict()

    copula_spec['u'] = [u_1, u_2]
    copula_spec['delta'] = delta
    copula_spec['r'] = r

    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'
    copula_spec['bounds'] = [200, 200]
    copula_spec['a'] = 1.0
    copula_spec['b'] = 0.0

    # We can now distribute the final specification for construction of the derived attributes.
    bounds = copula_spec['bounds']
    r = copula_spec['r']
    a = copula_spec['a']
    b = copula_spec['b']

    copula_spec['x_uniattribute_utility'] = PowerCls(r[0], a, b, bounds[0])
    copula_spec['y_uniattribute_utility'] = PowerCls(r[1], a, b, bounds[1])


    copula = UtilityCopulaCls(copula_spec)

    rslt = None

    if lottery == 1:
        rslt = 0.50 * copula.evaluate(15, 0) + 0.50 * copula.evaluate(20, 0)
    elif lottery == 2:
        rslt = 0.50 * copula.evaluate(30, 0) + 0.50 * copula.evaluate(40, 0)
    elif lottery == 3:
        rslt = 0.50 * copula.evaluate(60, 0) + 0.50 * copula.evaluate(80, 0)
    elif lottery == 4:
        rslt = 0.50 * copula.evaluate(0, 15) + 0.50 * copula.evaluate(0, 20)
    elif lottery == 5:
        rslt = 0.50 * copula.evaluate(0, 30) + 0.50 * copula.evaluate(0, 40)
    elif lottery == 6:
        rslt = 0.50 * copula.evaluate(0, 60) + 0.50 * copula.evaluate(0, 80)
    elif lottery == 7:
        rslt = 0.50 * copula.evaluate(15, 25) + 0.50 * copula.evaluate(25, 15)
    elif lottery == 8:
        rslt = 0.50 * copula.evaluate(30, 50) + 0.50 * copula.evaluate(50, 30)
    elif lottery == 9:
        rslt = 0.50 * copula.evaluate(60, 100) + 0.50 * copula.evaluate(100, 60)

    return rslt




def expected_utility_b(r, delta, u_1, u_2, lottery, m):

    copula_spec = dict()

    copula_spec['u'] = [u_1, u_2]
    copula_spec['delta'] = delta
    copula_spec['r'] = r

    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'
    copula_spec['bounds'] = [200, 200]
    copula_spec['a'] = 1.0
    copula_spec['b'] = 0.0

    # We can now distribute the final specification for construction of the derived attributes.
    bounds = copula_spec['bounds']
    r = copula_spec['r']
    a = copula_spec['a']
    b = copula_spec['b']

    copula_spec['x_uniattribute_utility'] = PowerCls(r[0], a, b, bounds[0])
    copula_spec['y_uniattribute_utility'] = PowerCls(r[1], a, b, bounds[1])


    copula = UtilityCopulaCls(copula_spec)


    rslt = None

    if lottery == 1:
        rslt = 0.50 * copula.evaluate(10 + m, 0) + 0.50 * copula.evaluate(25 + m, 0)
    elif lottery == 2:
        rslt = 0.50 * copula.evaluate(20 + m, 0) + 0.50 * copula.evaluate(50 + m, 0)
    elif lottery == 3:
        rslt = 0.50 * copula.evaluate(40 + m, 0) + 0.50 * copula.evaluate(100 + m, 0)
    elif lottery == 4:
        rslt = 0.50 * copula.evaluate(0, 10 + m) + 0.50 * copula.evaluate(0, 25 + m)
    elif lottery == 5:
        rslt = 0.50 * copula.evaluate(0, 20 + m) + 0.50 * copula.evaluate(0, 50 + m)
    elif lottery == 6:
        rslt = 0.50 * copula.evaluate(0, 40 + m) + 0.50 * copula.evaluate(0, 100 + m)
    elif lottery == 7:
        rslt = 0.50 * copula.evaluate(15 + m, 15) + 0.50 * copula.evaluate(25 + m, 25)
    elif lottery == 8:
        rslt = 0.50 * copula.evaluate(30 + m, 30) + 0.50 * copula.evaluate(50 + m, 50)
    elif lottery == 9:
        rslt = 0.50 * copula.evaluate(60 + m, 60) + 0.50 * copula.evaluate(100 + m, 100)
    return rslt



if __name__ == '__main__':
    from statsmodels.tools.eval_measures import rmse as get_rmse
    from scipy.optimize import minimize
    from scipy.stats import norm
    from scipy import optimize

    from functools import partial

    def comp_criterion_function(r, delta, u_1, u_2, lottery, version, m):
        """Criterion function for the root-finding function."""
        # TODO: This does in no way exploit single crossing ...
        stat_a = expected_utility_a(r, delta, u_1, u_2, lottery)
        stat_b = expected_utility_b(r, delta, u_1, u_2, lottery, m)

        if version == 'brenth':
            stat = stat_a - stat_b
        elif version == 'grid':
            stat = (stat_a - stat_b) ** 2
        else:
            raise NotImplementedError
        return stat
    count = 0
    while True:
        print(count)
        count += 1
        constr = dict()


        constr['r'] = [-0.1, -0.1]
        constr['delta'] = .001
        constr['u'] = 0.5, 0.49

        _, _, copula_spec = generate_random_request(constr)

        r = copula_spec['r']
        delta = copula_spec['delta']
        u_1, u_2 = copula_spec['u']





        #copula = UtilityCopulaCls(copula_spec)

        #try:
        from auxiliary_plots import create_contour_plot
        from auxiliary_plots import create_surface_plot

        #create_contour_plot(copula_spec, False, 'fig-contour')
        #create_surface_plot(copula_spec, False, 'fig-surface')

        #eu_1 = expected_utility_a(r, delta, u_1, u_2, lottery)
        #eu_2 = expected_utility_b(r, delta, u_1, u_2, lottery, 0.0)
        #print(eu_1, eu_2)
        crit_func = partial(comp_criterion_function, r, delta, u_1, u_2, lottery, 'brenth')
        m_opt = optimize.brenth(crit_func, 0, 100)
        print(m_opt)

        #
        # # TODO: This is probaly only relevant when r is risk seeking as then m < for a switch in
        # # sign required.
        # except ValueError:
        #     crit_func = partial(comp_criterion_function, r, delta, u_1, u_2, lottery, 'grid')
        #     crit_func = np.vectorize(crit_func)
        #     grid = np.linspace(0, 100, num=10, endpoint=True)
        #     m_opt = grid[np.argmin(crit_func(grid))]
        #
        break

