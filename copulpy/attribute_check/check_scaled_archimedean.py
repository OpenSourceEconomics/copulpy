"""Check attributes and properties of the scaled archimedean copula."""
import numpy as np

from copulpy.config_copulpy import IS_DEBUG


def check_attributes_scaled_archimedean(self, t=0, is_normalized=False):
    """Check the attributes of the class."""
    if not IS_DEBUG:
        return

    u_1, u_2 = self.get_attr('u_1', 'u_2')

    for u in [u_1, u_2]:
        np.testing.assert_equal(0 <= u <= 1, True)

    # Check input to evaluation
    np.testing.assert_equal(isinstance(t, int), True)
    np.testing.assert_equal(isinstance(is_normalized, (bool, np.bool_)), True)
