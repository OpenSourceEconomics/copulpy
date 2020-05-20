"""Provide the class that implements the nonstationary utility function."""
from functools import partial

import numpy as np

from copulpy.attribute_check.check_nonstationary import check_attributes_nonstationary
from copulpy.config_copulpy import HUGE_FLOAT
from copulpy.clsMeta import MetaCls


class NonstationaryUtilCls(MetaCls):
    """Manage the nonstationary utility function."""

    def __init__(self, alpha, beta, gamma, discount_factors, y_scale,
                 unrestricted_weights=None, discounting=None):
        """Initialize nonstationary utility function."""
        self.attr = dict()
        self.attr['y_scale'] = y_scale
        self.attr['alpha'] = alpha
        self.attr['gamma'] = gamma
        self.attr['beta'] = beta

        if discounting is not None:
            # Implement exponential discounting or hyperbolic discounting
            np.testing.assert_equal(discounting in ['exponential', 'hyperbolic'], True)

            if discounting in ['hyperbolic']:
                df_beta = discount_factors[0]
                df_delta = discount_factors[1]

                new_dfx = {
                    t: (df_beta * df_delta ** t if t > 0.0 else 1) for t in discount_factors.keys()
                }
            elif discounting in ['exponential']:
                df_delta = discount_factors[0]
                new_dfx = {t: (df_delta ** t if t > 0.0 else 1) for t in discount_factors.keys()}
            self.attr['discount_factors'] = new_dfx
        else:
            # Implement nonparametric discounting.
            self.attr['discount_factors'] = discount_factors

        # Optional argument: nonparametric weight on y_t in the CES function.
        if unrestricted_weights is None:
            # We apply the g() function here so that y_weights can be used identically below
            df = self.attr['discount_factors']
            y_weights = {t: y_scale * d_t ** (gamma - 1.0) for t, d_t in df.items()}
            self.attr['y_weights'] = y_weights
        else:
            # Nonparametric weight: no g() function applied in this case.
            self.attr['y_weights'] = unrestricted_weights

        self._check_attributes_nonstationary = partial(check_attributes_nonstationary, self)
        self._check_attributes_nonstationary()

    def evaluate(self, x, y, t=0):
        """Evaluate the flow utility from consumption (x,y) in period t."""
        alpha, beta, gamma, y_weights, discount_factors = \
            self.get_attr('alpha', 'beta', 'gamma', 'y_weights', 'discount_factors')
        # Marginals: power utility
        v_1 = x ** beta
        v_2 = y ** (beta * gamma)

        # Case distinction to avoid overflow error
        if x == 0.0:
            if y == 0.0:
                # Both zero
                utils = 0.0
            else:
                # Only y positive
                utils = discount_factors[t] * y_weights[t] * v_2
        else:
            if y == 0.0:
                # Only x positive
                utils = discount_factors[t] * v_1
            else:
                # Both positive.
                try:
                    utils = ((v_1 ** alpha) + ((y_weights[t] * v_2) ** alpha)) ** (1.0 / alpha)
                    utils = discount_factors[t] * utils
                # Sometimes an overflow error occurs.
                except ArithmeticError:
                    utils = HUGE_FLOAT

        return utils
