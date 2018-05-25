"""This module houses the Meta class for the package."""


class MetaCls(object):
    """This class collects all methods that are useful for all other classes in the package."""
    def _distribute_attributes(self, keys):
        """This method allows to quickly access all class attributes."""
        rslt = []
        for key_ in keys:
            rslt += [self.attr[key_]]

        # When only one attribute is requested, we do not need to return a list.
        if len(keys) == 1:
            rslt = rslt[0]

        return rslt
