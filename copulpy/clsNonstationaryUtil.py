"""Provide the class that implements the nonstationary utility function."""
import numpy as np

from copulpy.clsMeta import MetaCls


class NonstationaryUtilCls(MetaCls):
    """Manage the nonstationary utility function."""

    def __init__(self, alpha, beta, gamma, discont_factors, y_scale, restricted=True,
                 unrestricted_weights=None):
        """Initialize nonstationary utility function."""
        self.attr = dict()
        self.attr['alpha'] = alpha
        self.attr['beta'] = beta
        self.attr['gamma'] = gamma
        self.attr['discont_factors'] = discont_factors
        self.attr['y_scale'] = y_scale

        self._set_nonstationary_weights(restricted, unrestricted_weights)

        self._check_attributes()

    def evaluate(self, x, y, t=0):
        """Evaluate the flow utility from consumption (x,y) in period t."""
        alpha, beta, gamma = self.get_attr('alpha', 'beta', 'gamma')
        y_weights = self.get_attr('y_weights')
        discont_factors = self.get_attr('discont_factors')
        v_1 = x ** beta
        v_2 = y ** (beta * gamma)

        rslt = ((v_1 ** alpha) + (y_weights[t] ** alpha) * (v_2 ** alpha)) ** (1 / alpha)
        rslt = discont_factors[t] * rslt

        return rslt

    def _set_nonstationary_weights(self, restricted, unrestricted_weights):
        """Get the weight on y-utility in the CES function."""
        if restricted:
            discont_factors = self.get_attr('discont_factors')
            alpha, gamma, y_scale = self.get_attr('alpha', 'gamma', 'y_scale')

            y_weights = {t: y_scale * d_f ** (gamma - 1)
                         for t, d_f in discont_factors.items()}
            self.attr['y_weights'] = y_weights
        else:
            self.attr['y_weights'] = unrestricted_weights

    def _check_attributes(self):
        """Check attributes."""
        alpha, beta, gamma, y_scale, discont_factors, y_weights = \
            self.get_attr('alpha', 'beta', 'gamma', 'y_scale', 'discont_factors', 'y_weights')
        np.testing.assert_equal(0 <= alpha <= 1, True)
        np.testing.assert_equal(0 <= beta <= 1, True)
        np.testing.assert_equal(0 <= gamma <= 1, True)
        np.testing.assert_equal(0 <= y_scale, True)

        for period in [0, 1, 3, 6, 12, 24]:
            np.testing.assert_equal(0 <= discont_factors[period] <= 1, True)
            np.testing.assert_equal(0 <= y_weights[period], True)

    # Additional statistics for temporal decisions. In the future, this might be moved to trempy.
    def univariate_discont_factor(self, money, t):
        """Univariate discont factor."""
        beta = self.attr['beta']
        delta_t = self.attr['discont_factors'][t]

        indiff_amount = money * delta_t ** (-1 / beta)

        # These adjustment mirror Dennis' code which in turn is shadowing Thomas' Stata code
        ud_factor = (indiff_amount / money - 1) * 12 / max(t, 1)
        if (ud_factor != 0):
            ud_factor = ud_factor - 0.025

        if (ud_factor > 1.5):
            ud_factor = 1.525

        ud_factor = (1 / (1 + ud_factor * (t / 12)))

        return ud_factor

    def exchange_rate(self, money, t):
        """Intratemporal exchange rate."""
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_other = self.attr['y_scale']
        delta_t = self.attr['delta_t'][t]

        indiff_amount = ((money ** (1 / gamma)) * c_other ** (-beta * gamma) *
                         (delta_t ** ((1 - gamma) / (beta * gamma))))

        # This transformation is done in Dennis code, which in turn shadows Thomas' Stata code.
        ex_rate = (indiff_amount - 5) / money
        return ex_rate

    def multivariate_discont_factor_sc(self, money, t):
        """Multivariate discounting from self today to charity tomorrow."""
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_other = self.attr['y_scale']
        delta_t = self.attr['delta_t'][t]

        indiff_amount = ((c_other ** (-1 / (beta * gamma))) *
                         (delta_t ** (-1 / beta)) * (money ** (1 / gamma)))

        # This is again mirroring Dennis' code.
        md_sc = (indiff_amount - 3 * (1 + 1.5 * t / 12)) / money

        return md_sc

    def multivariate_discont_factor_cs(self, money, t):
        """Multivariate discounting from charity today to self tomorrow."""
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_other = self.attr['y_scale']
        delta_t = self.attr['delta_t'][t]
        assert (beta != 0) & (gamma != 0)
        indiff_amount = ((c_other ** (1 / beta)) * (delta_t ** (-1 / beta))) * (money ** gamma)

        # This is again mirroring Dennis' code.
        md_cs = (indiff_amount - (1 + 1.5 * t / 12)) / money
        return md_cs
