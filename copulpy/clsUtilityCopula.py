"""This module houses the class for the multiattribute utility copula."""
from functools import partial

import numpy as np

from copulpy.clsScaledArchimedean import ScaledArchimedeanCls
from copulpy.clsNonstationaryUtil import NonstationaryUtilCls
from copulpy.shared.auxiliary import distribute_copula_spec
from copulpy.clsWarmglowUtil import WarmglowUtilCls
from copulpy.clsExponential import ExponentialCls
from copulpy.config_copulpy import IS_DEBUG
from copulpy.clsPower import PowerCls
from copulpy.clsMeta import MetaCls

from copulpy.attribute_check.check_scaled_archimedean import check_attributes_scaled_archimedean
from copulpy.attribute_check.check_nonstationary import check_attributes_nonstationary
from copulpy.attribute_check.check_warmglow import check_attributes_warmglow

from copulpy.monitoring.monitoring_scaled_archimedean import log_scaled_archimedean
from copulpy.monitoring.monitoring_nonstationary import log_nonstationary
from copulpy.monitoring.monitoring_warmglow import log_warmglow


class UtilityCopulaCls(MetaCls):
    """Manage all things related to the multiattribute utility copulas."""

    def __init__(self, copula_spec):
        """Init class."""
        version = distribute_copula_spec(copula_spec, 'version')
        self.attr = dict()
        self.attr['version'] = version

        # Assign correct monitoring and attribute checks function.
        if version in ['scaled_archimedean']:
            self._logging = partial(log_scaled_archimedean, self)
            self._check_attributes = partial(check_attributes_scaled_archimedean, self)
        elif version in ['nonstationary']:
            self._logging = partial(log_nonstationary, self)
            self._check_attributes = partial(check_attributes_nonstationary, self)
        elif version in ['warmglow']:
            self._logging = partial(log_warmglow, self)
            self._check_attributes = partial(check_attributes_warmglow, self)
        else:
            raise NotImplementedError

        # Handle other attributes
        if version in ['scaled_archimedean']:
            args = ['r', 'bounds', 'delta', 'u', 'generating_function', 'a', 'b', 'marginals']
            r, bounds, delta, u, generating_function, a, b, marginals = \
                distribute_copula_spec(copula_spec, *args)

            marginal_utility = []
            for i, marginal in enumerate(marginals):
                if marginal == 'power':
                    marginal_utility += [PowerCls(r[i], a, b, bounds[i])]
                elif marginal == 'exponential':
                    marginal_utility += [ExponentialCls(r[i], a, b, bounds[i])]

            self.attr['x_uniattribute_utility'] = marginal_utility[0]
            self.attr['y_uniattribute_utility'] = marginal_utility[1]
            self.attr['bounds'] = bounds
            self.attr['delta'] = delta
            self.attr['u_1'] = u[0]
            self.attr['u_2'] = u[1]

            copula = ScaledArchimedeanCls(generating_function, u[0], u[1], delta)

        elif version in ['nonstationary', 'warmglow']:
            args = ['alpha', 'beta', 'gamma', 'discount_factors', 'y_scale',
                    'unrestricted_weights', 'discounting']
            alpha, beta, gamma, discount_factors, y_scale, unrestricted_weights, discounting = \
                distribute_copula_spec(copula_spec, *args)

            self.attr['unrestricted_weights'] = unrestricted_weights
            self.attr['discount_factors'] = discount_factors
            self.attr['discounting'] = discounting
            self.attr['y_scale'] = y_scale
            self.attr['alpha'] = alpha
            self.attr['gamma'] = gamma
            self.attr['beta'] = beta

            if version in ['nonstationary']:
                copula = NonstationaryUtilCls(
                    alpha, beta, gamma, discount_factors, y_scale, unrestricted_weights, discounting
                    )
            elif version in ['warmglow']:
                copula = WarmglowUtilCls(
                    alpha, beta, gamma, discount_factors, y_scale, unrestricted_weights, discounting
                    )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.attr['copula'] = copula
        self._check_attributes()
        self._logging()

    def evaluate(self, x, y, t=0, is_normalized=False):
        """Evaluate the multiattribute utility function."""
        attr = ['version', 'copula']
        version, copula = self.get_attr(attr)

        # Check integrity of class and request
        self._check_attributes()
        self._additional_checks(version, 'evaluate_in', x, y, t, is_normalized)

        if version in ['scaled_archimedean']:
            # Distribute class attributes
            attr = ['x_uniattribute_utility', 'y_uniattribute_utility']
            x_uniattribute_utility, y_uniattribute_utility = self.get_attr(attr)

            # Construct the normalized points of evaluation.
            if not is_normalized:
                v_1, v_2 = 0.0, 0.0
                if np.min(x) > 0:
                    v_1 = x_uniattribute_utility.evaluate(x, True)
                if np.min(y) > 0:
                    v_2 = y_uniattribute_utility.evaluate(y, True)
            else:
                v_1, v_2 = x, y

            rslt = copula.evaluate(v_1=v_1, v_2=v_2)

        elif version in ['nonstationary', 'warmglow']:
            rslt = copula.evaluate(x=x, y=y, t=t)
        else:
            raise NotImplementedError

        # Checks on return value
        self._additional_checks(version, 'evaluate_out', rslt)

        return rslt

    @staticmethod
    def _additional_checks(version, label, *args):
        """Perform some additional checks on selected features of the class instance."""
        # We only run these tests during debugging as otherwise the performance deteriorates.
        if not IS_DEBUG:
            return

        if label in ['evaluate_in']:
            x, y, t, is_normalized = args

            # General input checks
            np.testing.assert_equal(x >= 0, True)
            np.testing.assert_equal(y >= 0, True)

            # Version-specific input checks
            if version in ['scaled_archimedean']:
                np.testing.assert_equal(isinstance(is_normalized, (bool, np.bool_)), True)
            elif version in ['nonstationary', 'warmglow']:
                np.testing.assert_equal(isinstance(t, int), True)

        elif label in ['evaluate_out']:
            rslt, = args

            if version in ['scaled_archimedean']:
                np.testing.assert_equal(np.all(0.0 <= rslt) <= 1.0, True)
                np.testing.assert_equal(np.all(rslt <= 1.0), True)

            elif version in ['nonstationary', 'warmglow']:
                np.testing.assert_equal(np.all(rslt >= 0), True)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
