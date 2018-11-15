#!/usr/bin/env python
"""This module is a first take at regression tests."""
import pickle as pkl

import numpy as np

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.config_copulpy import PACKAGE_DIR


def transform_spec(copula_spec):
    """Map old format to new format."""
    new_spec = dict()
    version = copula_spec['version']
    new_spec['version'] = version
    new_spec[version] = dict()

    for key, value in copula_spec.items():
        new_spec[version][key] = value

    # check that mapping works.
    assert copula_spec.keys() == new_spec[version].keys()
    for key in copula_spec:
        if not isinstance(new_spec[version][key], np.ndarray):
            assert (new_spec[version][key] == copula_spec[key])
        else:
            assert (new_spec[version][key] == copula_spec[key]).all()
    return new_spec

# Generate tests for scaled archimedean copulas.
if False:
    NUM_TESTS = 1000
    tests = []
    for _ in range(NUM_TESTS):
        x, y, is_normalized, copula_spec = \
            generate_random_request({'version': 'scaled_archimedean'})
        rslt = UtilityCopulaCls(copula_spec).evaluate(x, y, t=0, is_normalized=is_normalized)
        tests += [[rslt, x, y, is_normalized, copula_spec]]

    pkl.dump(tests, open(PACKAGE_DIR + 'tests/regression_vault.copulpy.scaled_archimedean.pkl',
                         'wb'))

# Generate tests for nonstationary utility function.
if False:
    NUM_TESTS = 500
    tests = []
    for _ in range(NUM_TESTS):
        x, y, is_normalized, copula_spec = generate_random_request({'version': 'nonstationary'})
        for period in [0, 1, 3, 6, 12, 24]:
            rslt = UtilityCopulaCls(copula_spec).evaluate(x, y, t=period,
                                                          is_normalized=is_normalized)
            tests += [[rslt, x, y, period, is_normalized, copula_spec]]

    pkl.dump(tests, open(PACKAGE_DIR + '/tests/regression_vault.copulpy.nonstationary.pkl', 'wb'))

# Here, we decide which test to run.
current_test = 'nonstationary'

if current_test == 'scaled_archimedean':
    tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.copulpy.pkl', 'rb'))
    # tests = pkl.load(open('regression_vault.copulpy.pkl', 'rb'))

    totaltests = len(tests)
    counter = 0
    for test in tests:
        counter += 1
        print('Test {0} of {1} passed.'.format(counter, totaltests))
        rslt, x, y, is_normalized, old_copula_spec = test
        copula_spec = transform_spec(old_copula_spec)

        copula = UtilityCopulaCls(copula_spec)
        np.testing.assert_equal(copula.evaluate(x=x, y=y, is_normalized=is_normalized, t=0), rslt)

elif current_test == 'nonstationary':
    tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.copulpy.nonstationary.pkl', 'rb'))

    totaltests = len(tests)
    counter = 0
    for test in tests:
        counter += 1
        print('Test {0} of {1} passed.'.format(counter, totaltests))

        rslt, x, y, period, is_normalized, copula_spec = test
        copula = UtilityCopulaCls(copula_spec)
        np.testing.assert_equal(copula.evaluate(x=x, y=y, t=period,
                                                is_normalized=is_normalized), rslt)
