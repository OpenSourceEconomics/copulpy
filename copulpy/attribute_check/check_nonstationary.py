"""Check attributes and properties of the nonstationary utility function."""
import numpy as np


def check_attributes_nonstationary(self):
    """Check attributes."""
    alpha, beta, gamma, y_scale, discount_factors = \
        self.get_attr('alpha', 'beta', 'gamma', 'y_scale', 'discount_factors')
    np.testing.assert_equal(0.0 <= alpha <= 5.01, True)
    np.testing.assert_equal(0.0 <= beta <= 5.01, True)
    np.testing.assert_equal(0.0 <= gamma <= 5.01, True)
    np.testing.assert_equal(0.0 <= y_scale, True)
    for t in discount_factors.keys():
        np.testing.assert_equal(0.0 <= discount_factors[t] <= 1.01, True)
