import pymc as pm


class Model:
    '''
    CLAUDE: Base class of the parametric models fitted with PyMC (e.g. :class:`hydromass.model_mass.MassModel`, :class:`hydromass.model_abundance.Z_Ab_Model`).
    Subclasses set npar, parnames, func_pm, func_np and call :meth:`check` to store start, sd, limits, fix and prior.
    '''
    available_priors = ['Uniform', 'Normal', 'TruncatedNormal']

    def check(self, start = None, sd = None, limits = None, fix = None, prior = None):
        '''
        CLAUDE: Check the shapes of the prior settings and store them.
        '''
        assert len(start) == self.npar, 'Number of starting parameters does not match function.'
        assert len(sd) == self.npar, 'Shape of sd does not match function.'
        assert (limits.shape == (self.npar, 2)), 'Shape of limits does not match function.'
        assert (len(fix) == self.npar), 'Shape of fix vectory does not match function.'
        assert (len(prior) == self.npar), 'Shape of prior does not match function.'
        assert all(p in self.available_priors for p in prior), f'Unknown prior in {prior}. Available priors: {self.available_priors}.'

        self.start = start
        self.sd = sd
        self.limits = limits
        self.fix = fix
        self.prior = prior

    def pm_priors(self, logger = None):
        '''
        CLAUDE: Create the PyMC variables for the model parameters, starting at self.start. Must be called inside a pm.Model context.

        :param logger: Logger used to print the prior of each parameter. Default None.
        :return: list of PyMC variables, one per parameter, in the order of self.parnames
        '''
        pmod = []
        for i, name in enumerate(self.parnames):
            mu, sigma = self.start[i], self.sd[i]
            lower, upper = self.limits[i]
            if self.fix[i]:
                msg = f'pm.Deterministic(\'{name}\', pm.math.constant({mu}))'
                var = pm.Deterministic(name, pm.math.constant(mu))
            elif self.prior[i] == 'TruncatedNormal':
                msg = f'pm.TruncatedNormal(\'{name}\', mu={mu}, sigma={sigma}, lower={lower}, upper={upper})'
                var = pm.TruncatedNormal(name, mu = mu, sigma = sigma, lower = lower, upper = upper, initval = mu)
            elif self.prior[i] == 'Normal':
                msg = f'pm.Normal(\'{name}\', mu={mu}, sigma={sigma})'
                var = pm.Normal(name, mu = mu, sigma = sigma, initval = mu)
            else:
                msg = f'pm.Uniform(\'{name}\', lower={lower}, upper={upper})'
                var = pm.Uniform(name, lower = lower, upper = upper, initval = mu)
            if logger is not None:
                logger.info(f'  {msg}')
            pmod.append(var)
        return pmod
