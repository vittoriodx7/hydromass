import numpy as np
from scipy.optimize import minimize
import pymc as pm



def beta_Ab_pm(rad, norm, rc, beta, floor):
    return floor + norm * pm.math.exp(-beta * pm.math.log(1. + (rad / rc) ** 2))


def beta_Ab_np(rad, norm, rc, beta, floor):
    return floor + norm * np.exp(-beta * np.log(1. + (rad / rc) ** 2))


def constant_Ab_pm(rad, norm):
    return rad * 0 + norm


def constant_Ab_np(rad, norm):
    return rad * 0 + norm


class Z_Ab_Model:
    available_models = ['BETA_AB', 'CONSTANT_AB']

    @classmethod
    def help(cls):
        print(f'Available Abundance models: {cls.available_models}.')

    def __init__(self, model = 'BETA_AB', start = None, sd = None, limits = None, fix = None):

        if model not in self.available_models:
            raise ValueError(f'Unknown Iron model {model}. Available mass models: {self.available_models}.')

        if model == 'BETA_AB':
            self.setup_beta_Ab(start = start, sd = sd, limits = limits, fix = fix)
        elif model == 'CONSTANT_AB':
            self.setup_constant(start = start, sd = sd, limits = limits, fix = fix)

    def setup_beta_Ab(self, start = None, sd = None, limits = None, fix = None):
        self.func_pm = beta_Ab_pm
        self.func_np = beta_Ab_np
        self.npar = 4
        self.parnames = ['norm_ab', 'rc_ab', 'beta_ab', 'floor_ab']

        if start is None:
            start = [0.5, 30., 0.3, 0.35]
        if sd is None:
            sd = [1., 200., 5., 0.10]
        if limits is None:
            limits = np.empty((self.npar, 2))
            limits[0] = [-0.2, 20]
            limits[1] = [1, 2000]
            limits[2] = [-1, 10]
            limits[3] = [0, 1]
        if fix is None:
            fix = [False, False, True, False]

        self.check(start = start, sd = sd, limits = limits, fix = fix)

    def setup_constant(self, start = None, sd = None, limits = None, fix = None):
        self.func_pm = constant_Ab_pm
        self.func_np = constant_Ab_np
        self.npar = 1
        self.parnames = ['norm_ab']

        if start is None:
            start = [0.3]
        if sd is None:
            sd = [1.]
        if limits is None:
            limits = np.empty((self.npar, 2))
            limits[0] = [0, 3]

        if fix is None:
            fix = [False]

        self.check(start = start, sd = sd, limits = limits, fix = fix)

    def check(self, start = None, sd = None, limits = None, fix = None):

        assert len(start) == self.npar, 'Number of starting parameters does not match function.'
        assert len(sd) == self.npar, 'Shape of sd does not match function.'
        assert (limits.shape == (self.npar, 2)), 'Shape of limits does not match function.'
        assert (len(fix) == self.npar), 'Shape of fix vectory does not match function.'

        self.start = start
        self.sd = sd
        self.limits = limits
        self.fix = fix

    def quick_fit(self, SpecData):
        def optim_zfe(pars):
            c = 0
            for i in range(self.npar):
                if self.fix[i] == False:
                    if not (self.limits[i][0] < pars[c] < self.limits[i][1]):
                        return 1e10
                    c += 1

            c = 0
            pars_ab = []
            for i in range(self.npar):
                if self.fix[i] == False:
                    pars_ab.append(pars[c])
                    c += 1
                else:
                    pars_ab.append(self.start[i])

            pred = self.func_np(SpecData.rref_x_kpc, *pars_ab)

            chi2 = np.sum((pred - SpecData.zfe) ** 2 / (SpecData.zfe_hi ** 2 + 0.05 ** 2))

            return chi2

        pars_ab = []
        for i in range(self.npar):
            if self.fix[i] == False:
                pars_ab.append(self.start[i])

        res = minimize(optim_zfe, pars_ab, method = 'Nelder-Mead')

        c = 0
        for i in range(self.npar):
            if self.fix[i] == False:
                self.start[i] = float(res['x'][c])
                self.sd[i] = float(res['x'][c]) * 0.1
                c += 1