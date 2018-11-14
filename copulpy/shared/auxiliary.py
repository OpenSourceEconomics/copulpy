"""Auxiliary functions."""


def distribute_copula_spec(copula_spec, *keys):
    """Distribute the copula specification."""
    version = copula_spec['version']

    rslt = []
    for key_ in keys:
        rslt += [copula_spec[version][key_]]

    # Handle single arguments
    if len(rslt) == 1:
        rslt = rslt[0]

    return rslt
