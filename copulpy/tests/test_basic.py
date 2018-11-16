"""This module contains some tests for the development of sound multiattribute utility copulas."""
from subprocess import CalledProcessError
import pickle as pkl
import subprocess
import os

import numpy as np
import pytest

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.config_copulpy import PACKAGE_DIR


def test_1():
    """This test ensures that a multiattribute utility function is always created."""
    for _ in range(10):
        x, y, is_normalized, copula_spec = generate_random_request()
        copula = UtilityCopulaCls(copula_spec)

        periods = [0]
        if copula_spec['version'] == 'nonstationary':
            periods = copula_spec['nonstationary']['discount_factors'].keys()

        for period in periods:
            copula.evaluate(x, y, period, is_normalized)


@pytest.mark.skipif(os.getenv('CI') in ['true'], reason='slight numerical differences')
def test_2():
    """This test runs a subset of our regression vault."""
    tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.copulpy.pkl', 'rb'))
    for test in tests[:50]:
        rslt, x, y, period, is_normalized, copula_spec = test
        copula = UtilityCopulaCls(copula_spec)
        np.testing.assert_equal(copula.evaluate(x, y, period, is_normalized), rslt)


def test_3():
    """This test ensures that linear transformations of the uniattribute utility functions do not
    matter for the evaluation of the multiattribute utility copula."""
    constr = dict()
    constr['version'] = 'scaled_archimedean'
    period = np.random.randint(1, 100)

    for _ in range(10):
        x, y, is_normalized, copula_spec = generate_random_request(constr)

        copula = UtilityCopulaCls(copula_spec)
        base = copula.evaluate(x, y, period, is_normalized)

        for _ in range(10):
            copula_spec['a'] = np.random.uniform(0.01, 10)
            copula_spec['b'] = np.random.normal()

            copula = UtilityCopulaCls(copula_spec)
            np.testing.assert_almost_equal(base, copula.evaluate(x, y, period, is_normalized))


def test_4():
    """This test ensures that the basic conditions are satisfied by the multiattribute utility
    copula."""
    # TODO: Would it be useful to allow this kind of request for both copulas. This would require
    # the addition of the normalization feature to both, I guess.
    constr = dict()
    constr['version'] = 'scaled_archimedean'
    period = np.random.randint(1, 100)

    for _ in range(10):
        _, _, _, copula_spec = generate_random_request(constr)
        copula = UtilityCopulaCls(copula_spec)

        # Normalized range and domain
        np.testing.assert_equal(copula.evaluate(0, 0, period, True), 0.0)
        np.testing.assert_equal(copula.evaluate(1, 1, period, True), 1.0)

        # Nondecreasing in both arguments.
        v = np.random.uniform(0, 1, 2)
        base = copula.evaluate(*v, period, True)

        v_upper = []
        for item in v:
            v_upper += [np.random.uniform(item, 1)]
        np.testing.assert_equal(copula.evaluate(*v_upper, period, True) >= base, True)


def test_5():
    """This test runs flake8 to ensure the code quality. However, this is only relevant during
    development."""
    try:
        import flake8    # noqa: F401
    except ImportError:
        return None

    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR)
    try:
        subprocess.check_call(['flake8'])
        os.chdir(cwd)
    except CalledProcessError:
        os.chdir(cwd)
        raise AssertionError('... code does not conform to style guide')
