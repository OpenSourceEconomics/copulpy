
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

copula_spec = dict()

copula_spec['r'] = [-1.5, -0.5]
copula_spec['u'] = 0.1, 0.5
copula_spec['bounds'] = 10, 150
copula_spec['delta'] = 1.6

copula_spec['generating_function'] = 1
copula_spec['version'] = 'scaled_archimedean'
copula_spec['a'] = 10.0
copula_spec['b'] = 17.0

copula = UtilityCopulaCls(copula_spec)
u_eval = []
u_eval += [copula.evaluate(1, 0, True)]
u_eval += [copula.evaluate(0, 1, True)]
print(u_eval)
