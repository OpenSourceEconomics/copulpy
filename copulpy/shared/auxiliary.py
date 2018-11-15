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


def build_copula_spec(version, **kwargs):
    """Build a copula spec from labeled arguments."""
    spec = dict()
    spec['version'] = version
    spec[version] = dict()
    spec[version]['version'] = version

    if kwargs is not None:
        for key, value in kwargs.items():
            spec[version][key] = value

    # Handle all optional arguments here
    if (version == 'nonstationary') and ('unrestricted_weights' not in kwargs):
        spec['nonstationary']['unrestricted_weights'] = None

    return spec
