"""Auxiliary functions."""
import numpy as np


def distribute_copula_spec(copula_spec, *keys):
    """Distribute the copula specification."""
    rslt = []
    for key_ in keys:
        rslt += [copula_spec[key_]]

    return rslt


def process_consumption_bundle(bundle):
    """Process consumption bundles in various input formats."""
    if isinstance(bundle, list):
        # Nothing to do. It is already a list of lists
        return bundle

    elif isinstance(bundle, dict):
        # Convert dictionaries of the form {Time: (self money, other money),...} to list
        periods = list(bundle.keys())
        periods.sort()
        bundle_list = []
        for period in periods:
            bundle_list += [period, bundle[period][0], bundle[period][0]]

    elif isinstance(bundle, np.ndarray):
        sorted_bundle = bundle[bundle[:, 0].argsort()]
        bundle_list = list(sorted_bundle)
    else:
        raise Exception('Invalid consumption bundle.')
    return bundle_list
