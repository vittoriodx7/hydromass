import numpy as np
import pymc as pm

from .model import Model


def gnfw_logp0_pm(rad, logp0, c500, gamma, alfa, beta):
    '''
    CLAUDE: Generalized NFW pressure profile (Nagai et al. 2007), with the normalization given as logp0 = log(p0), for PyMC tensors

    .. math::

        P_{gNFW}(r) = \\frac{P_0} {(c_{500} r)^\\gamma (1+(c_{500} r)^\\alpha)^{(\\beta-\\gamma)/\\alpha}}

    computed in log space as log P = log P_0 - gamma log x - (beta - gamma) / alpha log(1 + x^alpha), with x = c500 r.

    :param rad: radius [kpc]
    :return: model pressure [keV cm^-3]
    '''
    log_x = pm.math.log(c500 * rad)
    return pm.math.exp(logp0 - gamma * log_x - (beta - gamma) / alfa * pm.math.log1pexp(alfa * log_x))


def gnfw_logp0_np(xout, pars):
    '''
    CLAUDE: Numpy version of :func:`gnfw_logp0_pm` for many parameter samples.

    :param xout: 1-D array of radii [kpc]
    :param pars: 2-D array of parameter samples (nsamples, 5), columns logp0, c500, gamma, alpha, beta
    :return: 2-D array of model pressure profiles (npoints, nsamples) [keV cm^-3]
    '''
    logp0, c500, gamma, alfa, beta = np.asarray(pars, dtype = float).T
    log_x = np.log(np.outer(xout, c500))
    return np.exp(logp0 - gamma * log_x - (beta - gamma) / alfa * np.logaddexp(0., alfa * log_x))


def gnfw_dlogp_dlogr_pm(rad, logp0, c500, gamma, alfa, beta):
    '''
    CLAUDE: Analytic logarithmic slope d ln P / d ln r of :func:`gnfw_logp0_pm`, for PyMC tensors (independent of logp0).
    Written as - (gamma + (beta - gamma) sigmoid(alpha ln x)), with x = c500 r, since x^alpha / (1 + x^alpha) = sigmoid(alpha ln x).

    :param rad: radius [kpc]
    :return: logarithmic slope of the model pressure
    '''
    # sigmoid (1 / (1 + exp(-x))
    return - (gamma + (beta - gamma) * pm.math.sigmoid(alfa * pm.math.log(c500 * rad)))


def gnfw_dlogp_dlogr_np(xout, pars):
    '''
    CLAUDE: Analytic logarithmic slope of the generalized NFW pressure profile for many parameter samples (independent of p0)

    .. math::

        \\frac{d \\ln P}{d \\ln r} = - \\left( \\gamma + \\frac{(\\beta - \\gamma)(c_{500}r)^{\\alpha}} {1 + (c_{500}r)^{\\alpha} } \\right)

    :param xout: 1-D array of radii [kpc]
    :param pars: 2-D array of parameter samples (nsamples, 5), columns logp0, c500, gamma, alpha, beta
    :return: 2-D array of slopes (npoints, nsamples)
    '''
    logp0, c500, gamma, alfa, beta = np.asarray(pars, dtype = float).T
    x_alfa = np.outer(xout, c500) ** alfa
    return - (gamma + (beta - gamma) * x_alfa / (1. + x_alfa))



class ForwardModel(Model):
    """
    Class allowing the user to define a parametric forward model to the gas pressure. Currently only supports the generalized NFW model (Nagai et al. 2007), :func:`gnfw_logp0_pm`.

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
        self.func_der_pm = gnfw_dlogp_dlogr_pm
        self.func_der_np = gnfw_dlogp_dlogr_np

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
