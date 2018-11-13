"""Provide the constant elasticity of substitution function."""
import numpy as np

from copulpy.config_copulpy import IS_DEBUG
from copulpy.clsMeta import MetaCls


class CESCls(MetaCls):
    """CES class."""

    def __init__(self, alpha, y_weight, discont_factor):
        """Initialize class."""
        self.attr = dict()

        self.attr['alpha'] = alpha
        self.attr['y_weight'] = y_weight
        self.attr['discont_factor'] = discont_factor

        self._check_attributes()

    def evaluate(self, v_1, v_2):
        """Evaluate the CES function."""
        self._additional_checks('evaluate_in', v_1, v_2)

        y_weight, discont_factor, alpha = self.get_attr('y_weight', 'discont_factor', 'alpha')

        rslt = (v_1 ** alpha + y_weight * v_2 ** alpha) ** (1 / alpha)
        rslt = discont_factor * rslt
        self._additional_checks('evaluate_out')

        return rslt

    def _check_attributes(self):
        """Check the attributes of the class."""
        y_weights = self.get_attr('y_weight')
        discont_factors = self.get_attr('discont_factor')
        alpha = self.get_attr('alpha')
        np.testing.assert_equal(alpha >= 0, True)
        np.testing.assert_equal(np.all(y_weights >= 0), True)
        np.testing.assert_equal(np.all(discont_factors >= 0), True)

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
            np.testing.assert_equal(np.all(0.0 <= rslt), True)
        else:
            raise NotImplementedError
