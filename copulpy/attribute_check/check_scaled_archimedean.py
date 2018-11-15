"""Check attributes and properties of the scaled archimedean copula."""
import numpy as np


def check_attributes_scaled_archimedean(self):
    """Check the attributes of the class."""
    u_1, u_2 = self.get_attr('u_1', 'u_2')

    for u in [u_1, u_2]:
        np.testing.assert_equal(0 <= u <= 1, True)
