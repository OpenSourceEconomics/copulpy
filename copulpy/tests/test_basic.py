"""This module contains some tests for the development of sound multiattribute utility copulas."""
import pickle as pkl

import numpy as np

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.config_copulpy import PACKAGE_DIR


def test_1():
    """This test ensures that a multiattribute utility function is always created."""
    for _ in range(10):
        x, y, is_normalized, copula_spec = generate_random_request()
        copula = UtilityCopulaCls(copula_spec)
        copula.evaluate(x, y, is_normalized)


def test_2():
    """This test runs a subset of our regression vault."""
    tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.copulpy.pkl', 'rb'))
    for test in tests[:50]:
        rslt, x, y, is_normalized, copula_spec = test
        copula = UtilityCopulaCls(copula_spec)
        np.testing.assert_equal(copula.evaluate(x, y, is_normalized), rslt)


def test_3():
    """This test ensures that linear transformations of the uniattribute utility functions do not
    matter for the evaluation of the multiattribute utility copula."""
    for _ in range(10):
        x, y, is_normalized, copula_spec = generate_random_request()

        copula = UtilityCopulaCls(copula_spec)
        base = copula.evaluate(x, y, is_normalized)

        for _ in range(10):
            copula_spec['a'] = np.random.uniform(0.01, 10)
            copula_spec['b'] = np.random.normal()

            copula = UtilityCopulaCls(copula_spec)
            np.testing.assert_equal(base, copula.evaluate(x, y, is_normalized))


def test_4():
    """This test ensures that the basic conditions are satisfied by the multiattribute utility
    copula."""
    for _ in range(10):
        _, _, _, copula_spec = generate_random_request()
        copula = UtilityCopulaCls(copula_spec)

        # Normalized range and domain
        np.testing.assert_equal(copula.evaluate(0, 0, True), 0.0)
        np.testing.assert_equal(copula.evaluate(1, 1, True), 1.0)

        # Nondecreasing in both arguments.
        v = np.random.uniform(0, 1, 2)
        base = copula.evaluate(*v, True)

        v_upper = []
        for item in v:
            v_upper += [np.random.uniform(item, 1)]
        np.testing.assert_equal(copula.evaluate(*v_upper, True) >= base, True)


def test_5():
    """This test ensures that the expected utility evaluation is in line with the specified risk
    preferences."""
    for _ in range(10):

        # We first specify a random request for the uniattribute utility function.
        attribute = np.random.choice(['x', 'y'])
        constr = dict()
        if np.random.choice([True, False], p=[0.2, 0.8]):
            constr['r'] = [1, 1]
        else:
            pass
        _, _, _, copula_spec = generate_random_request(constr)

        copula = UtilityCopulaCls(copula_spec)
        if attribute in ['x']:
            bound, col = copula_spec['bounds'][0], 0
        else:
            bound, col = copula_spec['bounds'][1], 1

        lower = np.random.uniform(0.05, bound * 0.5)
        upper = np.random.uniform(lower, bound * 0.75)
        spread = np.random.uniform(0.01, min(lower, bound - upper))

        if attribute in ['x']:
            eval_1 = [(lower, 0), (upper, 0)]
            eval_2 = [(lower - spread, 0), (upper + spread, 0)]
        else:
            eval_1 = [(0, lower), (0, upper)]
            eval_2 = [(0, lower - spread), (0, upper + spread)]

        rslt_1 = 0.50 * copula.evaluate(*eval_1[0]) + 0.50 * copula.evaluate(*eval_1[1])
        rslt_2 = 0.50 * copula.evaluate(*eval_2[0]) + 0.50 * copula.evaluate(*eval_2[1])

        # For very small evaluation points, the tested relationships do not hold with strict
        # inequalities,
        if (rslt_1 == 0.0) and (rslt_2 == 0.0) and (copula_spec['r'][col] != 1.0):
            continue

        if copula_spec['r'][col] == 1.0:
            # We cannot enforce a stricter inequality as the copula is actually fitted to
            # reproduce the uniattribute utility functions. This fit need not necessarily be
            # perfect.
            np.testing.assert_almost_equal(rslt_1, rslt_2, decimal=3)
        elif copula_spec['r'][col] > 1:
            np.testing.assert_equal(rslt_1 < rslt_2, True)
        else:
            np.testing.assert_equal(rslt_1 > rslt_2, True)
