#!/usr/bin/env python
"""This script is a basic property testing setup."""
from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls

for _ in range(1000):
    # Distribute request
    x, y, is_normalized, copula_spec = generate_random_request()
    version = copula_spec['version']

    # Handle versions
    if version in ['scaled_archimedean']:
        rslt = UtilityCopulaCls(copula_spec).evaluate(
            x=x, y=y, t=0, is_normalized=is_normalized
        )

    elif version in ['nonstationary']:
        for period in copula_spec['nonstationary']['discount_factors'].keys():
            rslt = UtilityCopulaCls(copula_spec).evaluate(
                x=x, y=y, t=period, is_normalized=is_normalized
            )
    else:
        raise NotImplementedError
