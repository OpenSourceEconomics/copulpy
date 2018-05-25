#!/usr/bin/env python
"""This module allows to explore the issues around using multiattribute utility copulas."""
import numpy as np

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.clsPower import PowerCls

from scipy import optimize

from functools import partial



def determine_optimal_compensation(r_self, r_other, delta, self, other, lottery):
    """This function determine the optimal compensation that ensures the equality of the expected
    utilities."""

    copula_spec = dict()


    copula_spec['r'] = [r_self, r_other]
    copula_spec['delta'] = delta
    copula_spec['u'] = self, other



    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'

    # TODO: These need to be integarated
    copula_spec['bounds'] = 200, 200
    copula_spec['a'] = 1.0
    copula_spec['b'] = 0.0

    # We can now distribute the final specification for construction of the derived attributes.
    bounds = copula_spec['bounds']
    r = copula_spec['r']
    a = copula_spec['a']
    b = copula_spec['b']

    # TODO: Why do we need to specify this here, and not internally?
    copula_spec['x_uniattribute_utility'] = PowerCls(r[0], a, b, bounds[0])
    copula_spec['y_uniattribute_utility'] = PowerCls(r[1], a, b, bounds[1])


    copula = UtilityCopulaCls(copula_spec)



    def comp_criterion_function(copula, lottery, version, m):
        """Criterion function for the root-finding function."""
        stat_a = expected_utility_a(copula, lottery)
        stat_b = expected_utility_b(copula, lottery, m)

        if version == 'brenth':
            stat = stat_a - stat_b
        elif version == 'grid':
            stat = (stat_a - stat_b) ** 2
        else:
            raise NotImplementedError
        return stat

    # For some parametrization our first choice fails as f(a) and f(b) must have different
    # signs. If that is the case, we use a simple grid search as backup.
    #try:
    crit_func = partial(comp_criterion_function, copula, lottery, 'brenth')
    m_opt = optimize.brenth(crit_func, 0.00, 100)
    #except ValueError:
    #crit_func = partial(comp_criterion_function, alpha, beta, eta, lottery, 'grid')
    #crit_func = np.vectorize(crit_func)
    #grid = np.linspace(0, 100, num=5000, endpoint=True)
    #m_opt = grid[np.argmin(crit_func(grid))]

    return m_opt

#determine_optimal_compensation(r_self, r_other, delta, self, other, lottery)

from trempy.shared.shared_auxiliary import expected_utility_a, expected_utility_b

for _ in range(1000):
    constr = dict()
    #constr['r'] = [r_self, r_other]
    #constr['delta'] = delta
    constr['bounds'] = np.random.random_integers(150, 200, 2)
    m = np.random.uniform(1, 5)
    _, _, copula_spec = generate_random_request(constr)

    copula = UtilityCopulaCls(copula_spec)

    lottery = np.random.choice(range(1, 16))
    expected_utility_a(copula, lottery)
    expected_utility_b(copula, lottery, m)


    r_self = 0.2
    r_other = -0.2
    delta = 1.0
    self, other = 0.5, 0.5

    determine_optimal_compensation(r_self, r_other, delta, self, other, lottery)