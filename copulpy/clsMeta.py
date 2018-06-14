"""This module houses the Meta class for the package."""


class MetaCls(object):
    """This class collects all methods that are useful for all other classes in the package."""
    def get_attr(self, *keys):
        """This method allows to quickly access all class attributes."""
        # We want to be able to pass in list of keys and keys directly.
        if isinstance(keys[0], list):
            keys = keys[0]

        rslt = []
        for key_ in keys:
            rslt += [self.attr[key_]]

        # When only one attribute is requested, we do not need to return a list.
        if len(keys) == 1:
            rslt = rslt[0]

        return rslt
