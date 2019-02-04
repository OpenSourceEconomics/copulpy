"""Check attributes and properties of the nonstationary utility function."""
import numpy as np

from copulpy.config_copulpy import IS_DEBUG


def check_attributes_nonstationary(self):
    """Check attributes."""
    if not IS_DEBUG:
        return

    alpha, beta, gamma, y_scale, discount_factors = \
        self.get_attr('alpha', 'beta', 'gamma', 'y_scale', 'discount_factors')
    np.testing.assert_equal(0.0 <= alpha <= 5.01, True)
    np.testing.assert_equal(0.0 <= beta <= 5.01, True)
    np.testing.assert_equal(0.0 <= gamma <= 5.01, True)
    np.testing.assert_equal(0.0 <= y_scale, True)

    for period in discount_factors.keys():
        np.testing.assert_equal(0.0 <= discount_factors[period] <= 1.01, True)
