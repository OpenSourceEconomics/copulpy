"""Provide the class that implements the warm glow utility function."""
from functools import partial

import numpy as np

from copulpy.attribute_check.check_warmglow import check_attributes_warmglow
from copulpy.config_copulpy import HUGE_FLOAT
from copulpy.clsMeta import MetaCls


class WarmglowUtilCls(MetaCls):
    """Manage the warm glow utility function."""

    def __init__(self, alpha, beta, gamma, discount_factors, y_scale,
                 unrestricted_weights=None, discounting=None):
        """Initialize warmglow utility function."""
        self.attr = dict()
        self.attr['y_scale'] = y_scale  # weight on utility from charity euro
        self.attr['alpha'] = alpha  # warm glow parameter
        self.attr['gamma'] = gamma  # correlation aversion
        self.attr['beta'] = beta  # risk aversion for self and charity euro

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
            df = self.attr['discount_factors']
            y_weights = {t: y_scale for t, d_t in df.items()}
            self.attr['y_weights'] = y_weights
        else:
            # Nonparametric weight: no g() function applied in this case.
            self.attr['y_weights'] = unrestricted_weights

        self._check_attributes_warmglow = partial(check_attributes_warmglow, self)
        self._check_attributes_warmglow()

    def evaluate(self, x, y, t=0):
        """Evaluate the flow utility from consumption (x,y) in period t."""
        alpha, beta, gamma, y_weights, discount_factors = \
            self.get_attr('alpha', 'beta', 'gamma', 'y_weights', 'discount_factors')
        # Marginals: power utility
        v_1 = x ** beta
        v_2 = y ** beta

        # CES: aggregate marginals
        try:
            utils = ((v_1 ** gamma) + ((y_weights[t] * v_2) ** gamma)) ** (1.0 / gamma)
            utils = discount_factors[t] * utils
            # Add warm glow utility
            if y > 0:
                utils = utils + alpha
        # Sometimes an overflow error occurs.
        except ArithmeticError:
            utils = HUGE_FLOAT

        return utils
