"""Provide the class that implements the nonstationary utility function."""
import numpy as np

from copulpy.clsMeta import MetaCls


class NonstationaryUtilCls(MetaCls):
    """Manage the nonstationary utility function."""

    def __init__(self, alpha, beta, gamma, discont_factors, y_scale):
        """Initialize nonstationary utility function."""
        self.attr = dict()
        self.attr['alpha'] = alpha
        self.attr['beta'] = beta
        self.attr['gamma'] = gamma
        self.attr['discont_factors'] = discont_factors
        self.attr['y_scale'] = y_scale

        self._fit()

        self._check_attributes()

    def evaluate(self, x, y, t=0):
        """Evaluate."""
        alpha, beta, gamma = self.get_attr('alpha', 'beta', 'gamma')
        y_weights = self.get_attr('y_weights')
        discont_factors = self.get_attr('discont_factors')
        v_1 = x ** (alpha * beta)
        v_2 = y ** (alpha * gamma)

        rslt = (v_1 + y_weights[t] * v_2) ** (1 / alpha)
        rslt = discont_factors[t] * rslt

        return rslt

    def _fit(self):
        """Fit implied parameters."""
        discont_factors = self.get_attr('discont_factors')
        alpha, gamma, y_scale = self.get_attr('alpha', 'gamma', 'y_scale')

        y_weights = {t: y_scale * d_f ** (alpha * (gamma - 1))
                     for t, d_f in discont_factors.items()}

        self.attr['y_weights'] = y_weights

    def _check_attributes(self):
        """Check attributes."""
        alpha, beta, gamma, y_scale = self.get_attr('alpha', 'beta', 'gamma', 'y_scale')
        np.testing.assert_equal(0 <= alpha <= 1, True)
        np.testing.assert_equal(0 <= beta <= 1, True)
        np.testing.assert_equal(0 <= gamma <= 1, True)
        np.testing.assert_equal(0 <= y_scale, True)
