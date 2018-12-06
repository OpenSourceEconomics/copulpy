"""Setup for pytest."""
import os

import pytest

from copulpy.clsUtilityCopula import UtilityCopulaCls     # noqa: F401
from copulpy.config_copulpy import PACKAGE_DIR


def test():
    """Run basic tests for the package."""
    base = os.getcwd()

    os.chdir(PACKAGE_DIR)
    pytest.main()

    os.chdir(base)
