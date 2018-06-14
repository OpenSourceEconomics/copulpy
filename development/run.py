
from subprocess import CalledProcessError
import pickle as pkl
import subprocess
import os


import numpy as np
import pytest

from copulpy.tests.test_auxiliary import generate_random_request
from copulpy.clsUtilityCopula import UtilityCopulaCls
from copulpy.config_copulpy import PACKAGE_DIR


for _ in range(10):

    x, y, is_normalized, copula_spec = generate_random_request()
    copula = UtilityCopulaCls(copula_spec)
    copula.evaluate(x, y, is_normalized)