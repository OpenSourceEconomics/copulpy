"""This module contains some auxiliary functions that ease our testing efforts."""
import numpy as np


def generate_random_request(constr=None):
    """This function generates a random request to evaluate a multiattribute utility function."""
    constr = constr or dict()
    copula_spec = dict()

    version = np.random.choice(['scaled_archimedean'])
    if version in ['scaled_archimedean']:
        generating_function = np.random.choice([1])

    marginals = np.random.choice(['power', 'exponential'], 2)

    copula_spec['generating_function'] = generating_function
    copula_spec['marginals'] = marginals
    copula_spec['version'] = version

    copula_spec['bounds'] = np.random.uniform(0.1, 10, 2)
    copula_spec['u'] = np.random.uniform(0.01, 0.99, 2)
    copula_spec['delta'] = np.random.uniform(0.001, 5)
    copula_spec['r'] = np.random.uniform(0.001, 5, 2)
    copula_spec['a'] = np.random.uniform(0.01, 10)
    copula_spec['b'] = np.random.normal()

    # We want to be able to request some constraint special cases.
    if 'r' in constr.keys():
        copula_spec['r'] = constr['r']
    if 'bounds' in constr.keys():
        copula_spec['bounds'] = constr['bounds']
    if 'a' in constr.keys():
        copula_spec['a'] = constr['a']
    if 'b' in constr.keys():
        copula_spec['b'] = constr['b']
    if 'u' in constr.keys():
        copula_spec['u'] = constr['u']
    if 'delta' in constr.keys():
        copula_spec['delta'] = constr['delta']

    # We can now distribute the final specification for construction of the derived attributes.
    bounds = copula_spec['bounds']

    # These are derived attributes and thus need to be created at the very end.
    is_normalized = np.random.choice([True, False])
    if is_normalized:
        x, y = np.random.uniform(0, 1, 2)
    else:
        x = np.random.uniform(0, bounds[0])
        y = np.random.uniform(0, bounds[1])

    return x, y, is_normalized, copula_spec
