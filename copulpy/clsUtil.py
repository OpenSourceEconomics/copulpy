"""Contains the class for the intertemporal utility function."""
from copulpy.clsMeta import MetaCls
from copulpy.auxiliary.auxiliary import process_consumption_bundle


class Util(MetaCls):
    """The flow utility function of the intertemporal utility function."""

    def __init__(self, flow_params):
        """Initialize the class."""
        self.attr = dict()
        self.attr['alpha'] = flow_params['alpha']
        self.attr['beta'] = flow_params['beta']
        self.attr['gamma'] = flow_params['gamma']
        self.attr['c_self'] = flow_params['c_self']
        self.attr['c_other'] = flow_params['c_other']
        self.attr['delta_t'] = flow_params['delta_t']

    def evaluate(self, period, money_self, money_other):
        """Flow utility from a (self, other) consumption bundle in period t."""
        alpha = self.attr['alpha']
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_self = self.attr['c_self']
        c_other = self.attr['c_other']
        delta_period = self.attr['delta_t'][period]

        # Univariate utility in period t
        self_utils = c_self * (money_self ** beta)
        other_utils = (delta_period ** (gamma - 1) * (c_self ** gamma) *
                       c_other * money_other ** (beta * gamma))

        # Discounted flow utility after applying the copula (in this case, a CES function)
        flow_utils = delta_period * ((self_utils ** alpha) + (other_utils ** alpha)) ** (1 / alpha)
        return flow_utils

    def evaluate_today(self, money_self, money_other):
        """Shortcut for evaluating the flow utility at t = 0."""
        return self.evaluate(0, money_self, money_other)

    def evaluate_stream(self, consumption_bundles):
        """Evaluate the utility function at a (self, other) consumption stream."""
        inter_utils = 0

        converted_bundle = process_consumption_bundle(consumption_bundles)

        # Sum flow utilities
        for bundle in converted_bundle:
            inter_utils = inter_utils + self.evaluate(bundle[0], bundle[1], bundle[2])
        return inter_utils

    def univariate_discont_factor(self, period, money):
        """Univariate discont factor."""
        beta = self.attr['beta']
        delta_period = self.attr['delta_t'][period]
        assert beta > 0

        indiff_amount = money * delta_period ** (-1 / beta)

        # These adjustment mirror Dennis' code which in turn is shadowing Thomas' Stata code
        ud_factor = (indiff_amount / money - 1) * 12 / max(period, 1)
        if (ud_factor != 0):
            ud_factor = ud_factor - 0.025

        if (ud_factor > 1.5):
            ud_factor = 1.525

        ud_factor = (1 / (1 + ud_factor * (period / 12)))

        return ud_factor

    def exchange_rate(self, period, self_money):
        """Intratemporal exchange rate."""
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_self = self.attr['c_self']
        c_other = self.attr['c_other']
        delta_period = self.attr['delta_t'][period]
        indiff_amount = ((c_self ** ((1 - gamma) / (beta * gamma))) * (self_money ** (1 / gamma)) *
                         c_other ** (-beta * gamma) *
                         (delta_period ** ((1 - gamma) / (beta * gamma))))

        # This transformation is done in Dennis code, which in turn shadows Thomas' Stata code.
        ex_rate = (indiff_amount - 5) / self_money
        return ex_rate

    def multivariate_discont_factor_sc(self, period, money):
        """Multivariate discounting from self today to charity tomorrow."""
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_self = self.attr['c_self']
        c_other = self.attr['c_other']
        delta_period = self.attr['delta_t'][period]
        assert (beta != 0) & (gamma != 0)
        indiff_amount = ((c_self ** ((1 - gamma) / (beta * gamma))) *
                         (c_other ** (-1 / (beta * gamma))) *
                         (delta_period ** (-1 / beta))) * (money ** (1 / gamma))

        # This is again mirroring Dennis' code.
        md_sc = (indiff_amount - 3 * (1 + 1.5 * (period) / 12)) / money

        return md_sc

    def multivariate_discont_factor_cs(self, period, money):
        """Multivariate discounting from charity today to self tomorrow."""
        beta = self.attr['beta']
        gamma = self.attr['gamma']
        c_self = self.attr['c_self']
        c_other = self.attr['c_other']
        delta_period = self.attr['delta_t'][period]
        assert (beta != 0) & (gamma != 0)
        indiff_amount = ((c_self ** ((gamma - 1) / beta)) * (c_other ** (1 / beta)) *
                         (delta_period ** (-1 / beta))) * (money ** gamma)

        # This is again mirroring Dennis' code.
        md_cs = (indiff_amount - (1 + 1.5 * (period) / 12)) / money
        return md_cs
