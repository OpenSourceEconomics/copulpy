#!/usr/bin/env python
"""This module is a first take at regression tests."""
import pickle as pkl
import numpy as np

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.config_copulpy import PACKAGE_DIR

# Generate a new regression vault
if False:
    NUM_TESTS = 1000
    tests = []
    for _ in range(NUM_TESTS):
        # Distribute copula spec
        x, y, is_normalized, copula_spec = generate_random_request()
        version = copula_spec['version']

        if version in ['scaled_archimedean']:
            rslt = UtilityCopulaCls(
                copula_spec).evaluate(x=x, y=y, t=0, is_normalized=is_normalized)
            tests += [[rslt, x, y, 0, is_normalized, copula_spec]]

        elif version in ['nonstationary', 'warmglow']:
            # Evaluate 'nonstationary' and 'warmglow' copula at all possible periods t
            for period in copula_spec[version]['discount_factors'].keys():
                rslt = UtilityCopulaCls(copula_spec).evaluate(
                    x=x, y=y, t=period, is_normalized=is_normalized)
                tests += [[rslt, x, y, period, is_normalized, copula_spec]]

        else:
            raise NotImplementedError

    pkl.dump(tests, open(PACKAGE_DIR + '/tests/regression_vault.copulpy.pkl', 'wb'))

# Run regression test
tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.copulpy.pkl', 'rb'))
totaltests = len(tests)
counter = 0

for test in tests:
    counter += 1

    rslt, x, y, period, is_normalized, copula_spec = test
    version = copula_spec['version']
    copula = UtilityCopulaCls(copula_spec)

    np.testing.assert_almost_equal(
        copula.evaluate(x=x, y=y, t=period, is_normalized=is_normalized), rslt)
