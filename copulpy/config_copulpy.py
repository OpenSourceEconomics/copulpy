"""This module contains configurations for the whole package."""
import os

HUGE_FLOAT = 10e+20

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

# I want to run all debugging tests in the development environment.
IS_DEBUG = False
if os.getenv('COPULPY_DEV') == 'TRUE':
    IS_DEBUG = True
