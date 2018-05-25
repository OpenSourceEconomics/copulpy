

def distribute_copula_spec(copula_spec, *keys):
    """This function distributes the copula specification."""
    rslt = []
    for key_ in keys:
        rslt += [copula_spec[key_]]

    return rslt
