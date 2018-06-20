"""This module houses the class for the power utility function."""
from numbers import Number
import numpy as np

from copulpy.config_copulpy import IS_DEBUG
from copulpy.clsMeta import MetaCls


class PowerCls(MetaCls):
    """This class manages the uniattribute utility function."""
    def __init__(self, r, a=1, b=0, upper_bound=None):

        self.attr = dict()
        self.attr['upper_bound'] = upper_bound
        self.attr['r'] = r
        self.attr['a'] = a
        self.attr['b'] = b

        self._check_attributes()

    def evaluate(self, x, is_normalized=False):
        """This method evaluates the specified power utility function."""
        # Check integrity of request and class instance.
        self._additional_checks('evaluate_in', x)
        self._check_attributes()

        # Distribute class attribute
        upper_bound = self.get_attr('upper_bound')

        if is_normalized:
            np.testing.assert_equal(upper_bound is None, False)
            numerator = self._power_utility(x) - self._power_utility(0)
            denominator = self._power_utility(upper_bound) - self._power_utility(0)
            u = numerator / denominator
        else:
            u = self._power_utility(x)

        self._additional_checks('evaluate_out', u)

        return u

    def _power_utility(self, x):
        """This method evaluates power utility."""
        r, a, b = self.get_attr('r', 'a', 'b')

        if r > 0:
            rslt = a * (x ** r) + b
        else:
            raise NotImplementedError

        return rslt

    def _check_attributes(self):
        """This function checks the attributes of the class."""
        r, a, b, upper_bound = self.get_attr('r', 'a', 'b', 'upper_bound')

        for var in [r, a, b, upper_bound]:
            np.testing.assert_equal(isinstance(var, Number), True)

        for var in [a, r]:
            np.testing.assert_equal(var > 0, True)

        if upper_bound is not None:
            np.testing.assert_equal(upper_bound > 0, True)

    def _additional_checks(self, label, *args):
        """This method performs some additional checks on selected features of the class
        instance."""
        # We only run these tests during debugging as otherwise the performance deteriorates.
        if not IS_DEBUG:
            return

        # Distribute class attributes
        upper_bound = self.get_attr('upper_bound')

        if label in ["evaluate_out"]:
            u, = args
            np.testing.assert_equal(np.isfinite(u), True)
        elif label in ['evaluate_in']:
            x, = args
            np.testing.assert_equal(x >= 0, True)
            np.testing.assert_equal(x <= upper_bound, True)
        else:
            raise NotImplementedError
