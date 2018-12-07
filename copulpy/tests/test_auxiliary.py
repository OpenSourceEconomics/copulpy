"""This module contains some auxiliary functions that ease our testing efforts."""
import numpy as np


def generate_random_request(constr=None):
    """Generate a random request to evaluate a multiattribute utility function."""
    constr = constr or dict()
    copula_spec = dict()

    # Handle copula version
    if 'version' in constr.keys():
        version = constr['version']
    else:
        version = np.random.choice(['scaled_archimedean', 'nonstationary'])
    copula_spec['version'] = version

    # Handle copula spec
    if version in ['scaled_archimedean']:
        copula_spec['scaled_archimedean'] = dict()
        copula_spec['scaled_archimedean']['version'] = 'scaled_archimedean'

        generating_function = np.random.choice([1])
        marginals = np.random.choice(['power', 'exponential'], 2)

        copula_spec['scaled_archimedean']['generating_function'] = generating_function
        copula_spec['scaled_archimedean']['marginals'] = marginals
        copula_spec['scaled_archimedean']['bounds'] = np.random.uniform(0.1, 10, 2)
        copula_spec['scaled_archimedean']['u'] = np.random.uniform(0.01, 0.99, 2)
        copula_spec['scaled_archimedean']['delta'] = np.random.uniform(0.001, 5)
        copula_spec['scaled_archimedean']['r'] = np.random.uniform(0.001, 5, 2)
        copula_spec['scaled_archimedean']['a'] = np.random.uniform(0.01, 10)
        copula_spec['scaled_archimedean']['b'] = np.random.normal()

        # We want to be able to request some constraint special cases.
        if 'r' in constr.keys():
            copula_spec['r'] = constr['r']
        if 'bounds' in constr.keys():
            copula_spec['scaled_archimedean']['bounds'] = constr['bounds']
        if 'a' in constr.keys():
            copula_spec['scaled_archimedean']['a'] = constr['a']
        if 'b' in constr.keys():
            copula_spec['scaled_archimedean']['b'] = constr['b']
        if 'u' in constr.keys():
            copula_spec['scaled_archimedean']['u'] = constr['u']
        if 'delta' in constr.keys():
            copula_spec['scaled_archimedean']['delta'] = constr['delta']

    if version in ['nonstationary']:
        copula_spec['nonstationary'] = dict()
        copula_spec['nonstationary']['version'] = 'nonstationary'
        copula_spec['nonstationary']['alpha'] = np.random.uniform(0.4, 1.0)
        copula_spec['nonstationary']['beta'] = np.random.uniform(0.4, 1.0)
        copula_spec['nonstationary']['gamma'] = np.random.uniform(0.4, 1.0)
        copula_spec['nonstationary']['y_scale'] = np.random.uniform(0.5, 10.0)
        copula_spec['nonstationary']['discount_factors'] = \
            {t: np.random.uniform(0.3, 1.0) for t in [0, 1, 3, 6, 12, 24]}

        random_weights = {t: np.random.uniform(0.3, 1.0) for t in [0, 1, 3, 6, 12, 24]}

        # Optional arguments
        copula_spec['nonstationary']['unrestricted_weights'] = \
            np.random.choice([random_weights, None], p=[0.9, 0.1])
        copula_spec['nonstationary']['discounting'] = \
            np.random.choice([None, 'hyperbolic', 'exponential'], p=[0.8, 0.1, 0.1])

    # These are derived attributes and thus need to be created at the very end.
    is_normalized = np.random.choice([True, False])
    if is_normalized:
        x, y = np.random.uniform(0, 1, 2)
    else:
        if version in ['scaled_archimedean']:
            bounds = copula_spec['scaled_archimedean']['bounds']
            x = np.random.uniform(0, bounds[0])
            y = np.random.uniform(0, bounds[1])
        elif version in ['nonstationary']:
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
        else:
            raise NotImplementedError

    return x, y, is_normalized, copula_spec
