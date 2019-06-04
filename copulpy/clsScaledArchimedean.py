"""This module houses the class for the scaled Archimedean copula."""
from functools import partial
from numbers import Number

from scipy.optimize import least_squares
import numpy as np

from copulpy.config_copulpy import IS_DEBUG
from copulpy.clsMeta import MetaCls


class ScaledArchimedeanCls(MetaCls):
    """This class manages all things related to the scaled Archimedean copula."""

    def __init__(self, generating_function, u_1, u_2, delta):
        """Init method."""
        self.attr = dict()

        self.attr['delta'] = delta
        self.attr['u_1'] = u_1
        self.attr['u_2'] = u_2

        # The generating functions are collected at the bottom of the model.
        if generating_function in [1]:
            self.inverse_generating_function = inverse_generating_function_1
            self.generating_function = generating_function_1
        else:
            raise NotImplementedError

        # Derived attributes
        self.attr['m'] = self._fit()

        self._check_attributes()

    def evaluate(self, v_1, v_2):
        """Evaluate the copula."""
        # Check request
        self._additional_checks('evaluate_in', v_1, v_2)

        # Distribute class attributes
        m, delta = self.get_attr('m', 'delta')

        # Construct auxiliary objects
        m_1, m_2 = m

        # Construct Archimedean copula
        numerator = 1.0
        numerator *= self.generating_function(delta, m_1 * v_1)
        numerator *= self.generating_function(delta, m_2 * v_2)
        numerator = self.inverse_generating_function(delta, numerator)

        denominator = 1.0
        denominator *= self.generating_function(delta, m_1)
        denominator *= self.generating_function(delta, m_2)
        denominator = self.inverse_generating_function(delta, denominator)

        rslt = numerator / denominator

        # Check return value
        self._additional_checks('evaluate_out', rslt)

        return rslt

    def _check_attributes(self):
        """Check the attributes of the class."""
        delta = self.get_attr('delta')
        np.testing.assert_equal(delta > 0, True)

    def _fit(self):
        """Fit the remaining parameters of the copula."""
        # Distribute class attributes
        u_1, u_2 = self.attr['u_1'], self.attr['u_2']

        # Solve for the share parameters
        m_1, m_2 = self.get_coefficients(u_1, u_2)

        # Checks for return values
        self._additional_checks('_fit_out', m_1, m_2)

        return m_1, m_2

    def _get_scale(self, m_1, m_2):
        """Determine the scale of the copula."""
        # Check request
        self._additional_checks('_get_scale_in', m_1, m_2)

        # Distribute class attributes
        delta = self.get_attr('delta')

        rslt = 1.0
        rslt *= self.generating_function(delta, m_1)
        rslt *= self.generating_function(delta, m_2)
        rslt = self.inverse_generating_function(delta, rslt) ** (-1)
        return rslt

    def criterion(self, u_1, u_2, x):
        """Criterion function for the copula construction."""
        m_1, m_2 = x

        a = self._get_scale(m_1, m_2)

        f = np.tile(0.0, 2)
        f[0] = a * m_1 - u_1
        f[1] = a * m_2 - u_2

        return f

    def get_coefficients(self, u_1, u_2):
        """Get coefficients for Archimedean copula."""
        criterion = partial(self.criterion, u_1, u_2)

        bounds = [[0.01, 0.01], [0.99, 0.99]]
        opt = least_squares(criterion, [0.5, 0.5], bounds=bounds)

        fmt_ = ' {:<10}    ' + '{:25.15f}' * 2 + '\n'
        with open('fit.copulpy.info', 'w') as outfile:
            outfile.write(' Copula Fitting\n\n')
            outfile.write(fmt_.format(*[' x'] + opt['x'].tolist()))
            outfile.write(fmt_.format(*[' fun'] + opt['fun'].tolist()))
            outfile.write('\n')

        return opt['x']

    @staticmethod
    def _additional_checks(label, *args):
        """Perform some additional checks on selected features of the class instance."""
        # We only run these tests during debugging as otherwise the performance deteriorates.
        if not IS_DEBUG:
            return

        if label in ['evaluate_in']:
            for var in args:
                np.testing.assert_equal(np.all(var >= 0), True)
        elif label in ['evaluate_out']:
            rslt, = args
            np.testing.assert_equal(np.all(0.0 <= rslt) <= 1.0, True)
            np.testing.assert_equal(np.all(rslt <= 1.0), True)
        elif label in ['_fit_out']:
            m_1, m_2 = args
            for var in [m_1, m_2]:
                np.testing.assert_equal(isinstance(var, Number), True)
                np.testing.assert_equal(0.0 <= var <= 1.0, True)
        elif label in ['evaluate_in']:
            v_1, v_2 = args
            for var in [v_1, v_2]:
                np.testing.assert_equal(np.all(0.0 <= var) <= 1.0, True)
                np.testing.assert_equal(np.all(var <= 1.0), True)
        elif label in ['evaluate_out']:
            rslt, = args
            np.testing.assert_equal(np.all(0.0 <= rslt), True)
            np.testing.assert_equal(np.all(rslt <= 1.0), True)
        elif label in ['_get_scale_in']:
            m_1, m_2 = args
            for var in [m_1, m_2]:
                np.testing.assert_equal(isinstance(var, Number), True)
                np.testing.assert_equal(0.0 <= var <= 1.0, True)
        else:
            raise NotImplementedError


# We collect the generating functions here.
def generating_function_1(delta, t):
    """Generating function that yields a multiplicative form."""
    # Check request
    np.testing.assert_equal(np.all(delta > 0), True)
    np.testing.assert_equal(np.all(t <= 1.0), True)
    np.testing.assert_equal(np.all(0.0 <= t), True)

    return 1.0 - t ** delta


def inverse_generating_function_1(delta, t):
    """Inverse of the copula generating function."""
    # Check request
    np.testing.assert_equal(np.all(delta > 0), True)
    np.testing.assert_equal(np.all(t <= 1.0), True)
    np.testing.assert_equal(np.all(0.0 <= t), True)

    return (1.0 - t) ** (1.0 / delta)
