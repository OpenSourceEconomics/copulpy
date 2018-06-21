#!/usr/bin/env python

from subprocess import CalledProcessError
import pickle as pkl
import subprocess
import os


import numpy as np
import pytest

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.config_copulpy import PACKAGE_DIR
np.random.seed(123)

for _ in range(1000):
    x, y, is_normalized, copula_spec = generate_random_request()

    print(is_normalized, 'out')
    copula = UtilityCopulaCls(copula_spec)
    copula.evaluate(x, y, is_normalized)
