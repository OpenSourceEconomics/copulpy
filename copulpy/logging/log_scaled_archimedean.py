"""Log properties of the scaled archimedean copula."""


def log_scaled_archimedean(self):
    """Provide some basic logging."""
    # Distribute class attributes
    u = self.attr['u_1'], self.attr['u_2']

    fmt_ = ' {:<10}    ' + '{:25.15f}' * 2 + '\n'
    with open('fit.copulpy.info', 'a') as outfile:
        outfile.write(' Boundary Values\n\n')
        outfile.write(fmt_.format(*[' requested'] + list(u)))
        line = [' fitted', self.evaluate(x=1, y=0, is_normalized=True, t=0),
                self.evaluate(x=0, y=1, is_normalized=True, t=0)]
        outfile.write(fmt_.format(*line))
