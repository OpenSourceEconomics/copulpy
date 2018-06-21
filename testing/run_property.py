#!/usr/bin/env python
"""This script is a basic property testing setup."""
from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls

for _ in range(1000):
    x, y, is_normalized, copula_spec = generate_random_request()
    rslt = UtilityCopulaCls(copula_spec).evaluate(x, y, is_normalized)
