"""Check attributes and properties of the warmglow utility function."""
import numpy as np

from copulpy.config_copulpy import IS_DEBUG


def check_attributes_warmglow(self):
    """Check attributes."""
    if not IS_DEBUG:
        return

    alpha, beta, gamma, y_scale, discount_factors = \
        self.get_attr('alpha', 'beta', 'gamma', 'y_scale', 'discount_factors')

    cond = (0.01 <= alpha) & (0.01 <= beta) & (0.01 <= gamma) & (0.01 <= y_scale)
    np.testing.assert_equal(cond, True)

    cond = [(0.0 <= discount_factors[t] <= 1.01) for t in discount_factors.keys()]
    np.testing.assert_equal(np.all(cond), True)
