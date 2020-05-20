#!/usr/bin/env python
"""This script is a basic property testing setup."""
from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls

for _ in range(10000):

    x, y, is_normalized, copula_spec = generate_random_request()
    version = copula_spec['version']

    if version in ['nonstationary', 'warmglow']:
        periods = copula_spec[version]['discount_factors'].keys()
    else:
        periods = [0]

    for period in periods:
        UtilityCopulaCls(copula_spec).evaluate(x, y, period, is_normalized)
