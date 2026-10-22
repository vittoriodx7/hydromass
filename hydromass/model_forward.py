import numpy as np
import pymc as pm

from .model import Model
from .forward import gnfw_pm, gnfw_np, der_lnP_np


def gnfw_logp0_pm(rad, logp0, c500, gamma, alfa, beta):
    '''
    CLAUDE: Same as :func:`hydromass.forward.gnfw_pm`, with the normalization given as logp0 = log(p0).
    '''
    return gnfw_pm(rad, pm.math.exp(logp0), c500, gamma, alfa, beta)


def gnfw_logp0_np(xout, pars):
    '''
    CLAUDE: Same as :func:`hydromass.forward.gnfw_np`, with the first column of pars given as logp0 = log(p0).
    '''
    pars_lin = np.array(pars, dtype = float)
    pars_lin[:, 0] = np.exp(pars_lin[:, 0])
    return gnfw_np(xout, pars_lin)


class Forward(Model):
    """
    Class allowing the user to define a parametric forward model to the gas pressure. Currently only supports the generalized NFW model (Nagai et al. 2007), :func:`hydromass.forward.gnfw_pm`.

    The normalization is fitted in log space: the first parameter is logp0 = log(p0), and start, sd and limits of that parameter are given in log space.

    :param start: 1-D array including the central values of the Gaussian priors on the gNFW model parameters. If None, the starting values are set automatically using the average gNFW model of Planck Collaboration V (2013). Defaults to None.
    :type start: numpy.ndarray
    :param sd: 1-D array including the standard deviation values of the Gaussian priors on the gNFW model parameters. If None, the standard deviations are set automatically to encompass the variety of pressure profiles of Planck Collaboration V (2013). Defaults to None.
    :type sd: numpy.ndarray
    :param limits: 2-D array including the minimum and maximum allowed values for each gNFW parameter. If None, very broad automatic boundaries are used. Defaults to None.
    :type limits: numpy.ndarray
    :param fix: 1-D array of booleans describing whether each parameter is fitted (False) or fixed to the input value given by the "start" parameter (True). If None all the parameters are fitted. Defaults to None.
    :type fix: numpy.ndarray
    :param prior: List setting the prior type of each parameter, one of 'Uniform' (between limits), 'Normal' (start, sd) or 'TruncatedNormal' (start, sd, between limits). If None, all the parameters use 'TruncatedNormal'. Defaults to None.
    :type prior: list
    """
    def __init__(self, start=None, sd=None, limits=None, fix=None, prior=None):

        self.mod_type = 'FORWARD'

        self.func_np = gnfw_logp0_np
        self.func_pm = gnfw_logp0_pm
        self.func_der = der_lnP_np

        self.npar = 5
        self.parnames = ['logp0', 'c500', 'gamma', 'alpha', 'beta']

        if start is None:
            start = [np.log(1e-2), 1.5/1200., 0.3, 1.3, 4.4] # similar to Universal profile from Ghirardini+19
        if sd is None:
            sd = [1e-2 / 1e-2, 1.0/1200., 0.5, 0.5, 2.] # sd of logp0 is sd(p0) / p0
        if limits is None:
            limits = np.empty((self.npar, 2))
            limits[0] = [np.log(1e-5), np.log(1.)]
            limits[1] = [0.1/2000., 6./500.]
            limits[2] = [0., 3.]
            limits[3] = [0.5, 2.5]
            limits[4] = [1., 10.]
        if fix is None:
            fix = [False, False, False, False, False]

        if prior is None:
            prior = ['TruncatedNormal'] * self.npar

        self.check(start = start, sd = sd, limits = limits, fix = fix, prior = prior)
