import random
from curses import has_key
from logging import Logger
import shutil
import attrs
import ipdb
import numpy as np
from tqdm import tqdm
import os
from itertools import product
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from jax import random as jrandom
import matplotlib.pyplot as pl
import pytensor.tensor as tt
from jax import config
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
import nlopt

config.update("jax_enable_x64", True)
from .pyutils import create_logger

__all__ = ['is_xspec_available', 'calc_emissivity']


def is_xspec_available():
    # 1. Check if the executable is in the PATH
    executable_exists = shutil.which('xspec') is not None

    # 2. Check if HEASOFT environment variables are set
    # XSPEC usually needs HEADAS to function correctly
    env_initialized = "HEADAS" in os.environ

    return executable_exists and env_initialized


def vikh_temp_average(R, pars):
    T0 = pars[0]
    R500 = pars[1]
    rcool = pars[2]
    acool = pars[3]
    Tmin_over_T0 = pars[4]
    c = pars[5]
    rt = pars[6]
    b = 2
    a = 0
    #
    r = R / R500
    #
    x = (r / rcool) ** acool
    t1 = (Tmin_over_T0 + x) / (1 + x)
    t2 = (r / rt) ** (-a) / ((1 + (r / rt) ** b) ** c / b)
    return T0 * t1 * t2


def calc_emissivity(cosmo, z, nh, kt, Z, rmf, abund = 'aspl', elow = 0.5, ehigh = 2.0, unit = 'cr',
                    lum_elow = 0.5, lum_ehigh = 2.0, arf = None, tmpdir = None, quiet = True, absorption_model = 'phabs'):
    """

    Function calc_emissivity, computes scaling factor between count rate and APEC/MEKAL norm using XSPEC. The tool performs an XSPEC simulation of an absorbed APEC model with parameters set by the user, and computes the expected count rate in the band of interest corresponding to an APEC normalization of unity. The corresponding number is then used as conversion between count rate and emission measure for the hydrostatic mass reconstruction step. Note that since the computed count rates are corrected for vignetting through the exposure map, the provided ARF should correspond to the on-axis (unvignetted) effective area.
    Requires that XSPEC be in path

    :param cosmo: Astropy cosmology object containing the definition of the used cosmology.
    :type cosmo: class:`astropy.cosmology`
    :param z: Source redshift
    :type z: float
    :param nh: Source NH in units of 1e22 cm**(-2)
    :type nh: float
    :param kt: Source temperature in keV
    :type kt: float
    :param rmf: Path to response file (RMF/RSP)
    :type rmf: str
    :param abund: Solar abundance table in XSPEC format. Defaults to "aspl" (Asplund et al. 2009)
    :type abund: str
    :param Z: Metallicity with respect to solar. Defaults to 0.3
    :type Z: float
    :param elow: Low-energy bound of the input image in keV. Defaults to 0.5
    :type elow: float
    :param ehigh: High-energy bound of the input image in keV. Defaults to 2.0
    :type ehigh: float
    :param arf: Path to on-axis ARF (optional, in case response file is RMF)
    :type arf: str
    :param unit: Specify whether the exposure map is in units of sec (unit='cr') or photon flux (unit='photon'). By default unit='cr'.
    :type unit: str
    :param lum_elow: Low energy bound (rest frame) for luminosity calculation. Defaults to 0.5
    :type lum_elow: float
    :param lum_ehigh: High energy bound (rest frame) for luminosity calculation. Defaults to 2.0
    :type lum_ehigh: float
    :param quiet: Do not print all xspec output
    :type quiet: bool
    :return: Conversion factor
    :rtype: float
    """
    logger = create_logger('calc_emissivity')

    if unit not in ['cr', 'photon']:
        raise ValueError(f'Unknown {unit=}, aborting')

    check_xspec = is_xspec_available()
    if not check_xspec:
        raise ValueError('XSPEC cannot be found in path')

    if not os.path.isfile(rmf):
        raise ValueError('RMF file not found, aborting')

    if arf is not None and not os.path.exists(arf):
        raise ValueError(f'ARF file {arf} not found, aborting')

    purge_tmpdir = False
    if tmpdir is None:
        tmpdir = f'emissivity_tmp{random.randint(0, 1000)}'
        purge_tmpdir = True
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)

    pwd = os.getcwd()

    os.chdir(tmpdir)

    H0 = cosmo.H0.value
    Ode = cosmo.Ode0

    fakfile = 'sim.fak'
    if os.path.isfile(fakfile):
        os.remove(fakfile)

    with open('commands.xcm', 'w') as fsim:
        print('query y', file = fsim)
        print(f'cosmo {H0} 0 {Ode}', file = fsim)
        print(f'abund {abund}', file = fsim)
        print(f'model {absorption_model}(apec)', file = fsim)
        print(f'{nh}', file = fsim)  # nH
        print(f'{kt}', file = fsim)  # kT
        print(f'{Z}', file = fsim)  # metallicity
        print(f'{z}', file = fsim)  # redshift
        print('1.0', file = fsim)  # APEC norm

        print('fakeit none', file = fsim)
        print(f'{rmf}', file = fsim)
        if arf is not None:
            print(f'{arf}', file = fsim)
        else:
            print('', file = fsim)
        print('', file = fsim)
        print('', file = fsim)
        print(f'{fakfile}', file = fsim)
        print('10000, 1', file = fsim)
        print(f'ign **-{elow:1.2f} {ehigh:1.2f}-**', file = fsim)
        print(f'log sim.txt', file = fsim)
        if unit == 'cr':
            print('show rates', file = fsim)
            print('newpar 1 0 -1 0 0 1 1 ', file = fsim)
        print(f'flux {elow:1.2f} {ehigh:1.2f}', file = fsim)
        print('log none', file = fsim)
        print('delcomp 1', file = fsim)
        print(f'log lumin.txt', file = fsim)
        print(f'lumin {lum_elow:1.2f} {lum_ehigh:1.2f} {z}', file = fsim)
        print('log none', file = fsim)
        print('quit', file = fsim)

    if quiet:
        os.system(f'xspec - commands.xcm > /dev/null 2>&1')
    else:
        os.system(f'xspec - commands.xcm')

    with open(f'sim.txt') as ssim:
        lsim = ssim.readlines()

    if unit == 'cr':
        for line in lsim:
            if 'Model predicted rate:' in line:
                cr = float(line.split()[4])
    else:
        for line in lsim:
            if 'photons' in line:
                cr = float(line.split()[3])

    for line in lsim:
        if 'Model Flux' in line:
            flux = float(line.split()[5].replace('(', ''))

    slum = os.popen(f'grep Luminosity lumin.txt', 'r')
    llum = slum.readline()
    lumtot = float(llum.split()[2])
    lumfact = lumtot / cr

    os.chdir(pwd)

    if purge_tmpdir:
        shutil.rmtree(tmpdir)

    return cr, lumfact, lumtot, cr, flux



@attrs.define(kw_only = True, slots = False)
class ConversionFactor:
    logger: Logger = attrs.field(validator = attrs.validators.instance_of(Logger), init = False)

    absorption_model: str = attrs.field(validator = attrs.validators.instance_of(str), init = True)
    quiet: bool = attrs.field(validator = attrs.validators.instance_of(bool), init = True)
    conversion_factor_type: str = attrs.field(validator = attrs.validators.instance_of(str), init = True)

    table_options: dict = attrs.field(validator = attrs.validators.instance_of(dict), init = False, default=dict())


    def __attrs_post_init__(self):
        self.logger = create_logger('ConversionFactor')
        self.logger.info(self)

    @classmethod
    def from_mhyd(cls, mhyd, absorption_model, quiet, conversion_factor_type, lum_elow = 0.5, lum_ehigh = 2.0, table_options=None):
        cf = cls(
            absorption_model = absorption_model,
            quiet = quiet,
            conversion_factor_type = conversion_factor_type,
        )
        if conversion_factor_type == 'table':
            if table_options is None:
                table_options = {
                    'log_kt_min': -1.6,
                    'log_kt_max': np.exp(1),
                    'n_kt': 25,
                    'Z_ab_min': 0.05,
                    'Z_ab_max': 1.,
                    'n_Z_ab': 10
                }
            cf.conversion_factor_table(mhyd, lum_elow = lum_elow, lum_ehigh = lum_ehigh, **table_options)
        elif conversion_factor_type == 'single':
            cf.conversion_factor_single(mhyd, lum_elow = lum_elow, lum_ehigh = lum_ehigh)
        elif conversion_factor_type == 'variable':
            cf.conversion_factor_variable(mhyd, lum_elow = lum_elow, lum_ehigh = lum_ehigh)
        else:
            raise ValueError(f'Unknown conversion_factor_type: {conversion_factor_type}. Use \'table\', \'single\', or \'variable\'.')
        return cf

    def conversion_factor_single(self, Mhyd, lum_elow = 0.5, lum_ehigh = 2.0):
        # Get average temperature and metallicity from all spectral data
        kt = float(np.mean([np.average(this_spec_data.temp_x, weights = 1. / this_spec_data.errt_x ** 2) for this_spec_data in Mhyd.spec_data]))
        Z_ab = float(np.mean([np.average(this_spec_data.zfe, weights = 1. / this_spec_data.errt_zfe ** 2) for this_spec_data in Mhyd.spec_data]))

        for this_sb_data in Mhyd.sb_data:
            ccf, lf, _, _, _ = calc_emissivity(Mhyd.cosmo, Mhyd.redshift, Mhyd.nh, kt, Z_ab, this_sb_data.rmf, abund = Mhyd.abund, arf = this_sb_data.arf,
                                           elow = this_sb_data.elow, ehigh = this_sb_data.ehigh, unit = this_sb_data.unit, absorption_model = self.absorption_model, quiet = self.quiet, lum_elow = lum_elow, lum_ehigh = lum_ehigh)

            self.logger.info(f'{kt=:.2f}, {Z_ab=:.2f}: {ccf=}')

            this_sb_data.ccf, this_sb_data.llf = np.atleast_1d(ccf), np.atleast_1d(lf)

            this_sb_data.mean_cf = ccf


    def fit_kt(self, Mhyd):

        R500 = 600

        T0 = np.mean([np.average(this_spec_data.temp_x, weights = 1. / this_spec_data.errt_x ** 2) for this_spec_data in Mhyd.spec_data])

        pars_temp = np.array([T0, R500, 0.045, 1.9, 0.45, 0.9, 0.6])

        def optim_kt(facts, *args):

            tpars = np.exp(facts) * pars_temp

            chi2 = 0

            for this_spec_data in Mhyd.spec_data:

                mod_kt = vikh_temp_average(this_spec_data.rref_x_kpc, tpars)

                chi2 += np.sum((mod_kt - this_spec_data.temp_x) ** 2 / this_spec_data.errt_x ** 2)

            prior = np.sum(0.5 * ((facts - 0) / 1) ** 2)  # Don't change much from V06 values

            # if np.any(facts < 0.):
            #
            #     chi2 = chi2 + 1e10

            return chi2 + prior

        facts_temp = [0, 0, 0, 0, 0, 0, 0]

        lbounds = [-10, -10, -10, -10, -10, -10, -10]

        ubounds = [10, 10, 10, 10, 10, 10, 10]

        bounds = [(l, u) for l, u in zip(lbounds, ubounds)]

        res = minimize(optim_kt, facts_temp, method = 'Nelder-Mead', options = {'maxfev': 5000, 'maxiter': 5000})

        fpars = res['x']

        for i, (p, l, u) in enumerate(zip(fpars, lbounds, ubounds)):
            if p < l:
                fpars[i] = l

            if p > u:
                fpars[i] = u

        self.logger.info(res)

        methods = ('LN_SBPLX', 'LN_BOBYQA')  # 'LN_NELDERMEAD' , 'LN_COBYLA'

        flike = res.fun
        maxloops = 10
        sigdelta = 0.01
        initlike = optim_kt(fpars)
        for i in range(maxloops):
            self.logger.info('Fitting (iteration %i)' % (i + 1))

            for method_nlopt in methods:
                # nlopt methods
                meth = getattr(nlopt, method_nlopt.upper())
                opt = nlopt.opt(meth, len(fpars))
                opt.set_min_objective(optim_kt)
                opt.set_xtol_abs(1e-3)
                opt.set_lower_bounds(lbounds)
                opt.set_upper_bounds(ubounds)
                try:
                    fpars = opt.optimize(fpars)
                    flike = -opt.last_optimum_value()
                    num_evals = opt.get_numevals()
                    self.logger.info(f'Using method {method_nlopt}...{num_evals} steps')
                except ValueError as e:
                    self.logger.info('  Optimization failed: "%s"' % str(e))

            if abs(initlike - flike) < sigdelta:
                # fit quality stayed the same
                break
            initlike = flike
        else:
            self.logger.info('Warning: might be a bad fit, success = False')

        return fpars, pars_temp

    def fit_Z_ab(self, Mhyd):
        self.logger.info('Modeling abundance profile...')

        active_s = []
        rads_z_s = []
        for this_spec_data in Mhyd.spec_data:
            active = np.where(this_spec_data.zfe_lo > 0.)
            rads_z = this_spec_data.rref_x_kpc[active]

            active_s.append(active)
            rads_z_s.append(rads_z)

        beta_fe = 0.3

        pars_fe = np.array([30., 0.5, 0.2])

        def optim_zfe(pars):
            rc = pars[0]
            norm = pars[1]
            floor = pars[2]


            chi2=0

            for this_spec_data, active, rads_z in zip(Mhyd.spec_data, active_s, rads_z_s):
                pred = floor + norm * (1. + (rads_z / rc) ** 2) ** (-beta_fe)
                chi2 += np.sum((pred - this_spec_data.zfe[active]) ** 2 / this_spec_data.zfe_hi[active] ** 2)

            if rc <= 0 or floor < 0 or norm < -0.2:
                chi2 = chi2 + 1e10

            return chi2

        res = minimize(optim_zfe, pars_fe, method = 'Nelder-Mead')

        zpars = res['x']

        return zpars, beta_fe

    def conversion_factor_variable(self, Mhyd, lum_elow = 0.5, lum_ehigh = 2.0):

        for i_sb, this_sb_data in enumerate(Mhyd.sb_data):
            out_cfact_file = f'{Mhyd.dir}/emissivity_{i_sb}.dat'
            out_cfact_pdf = f'{Mhyd.dir}/emissivity_{i_sb}.pdf'

            if not os.path.isfile(out_cfact_file) or not os.path.isfile(out_cfact_pdf):

                fpars, pars_temp = self.fit_kt(Mhyd)
                zpars, beta_fe = self.fit_Z_ab(Mhyd)

                kt_prof = vikh_temp_average(this_sb_data.rref_x_kpc, np.exp(fpars) * pars_temp)
                Z_ab_prof = zpars[2] + zpars[1] * (1. + (this_sb_data.rref_x_kpc / zpars[0]) ** 2) ** (-beta_fe)


                cf_prof = np.zeros(len(kt_prof))
                lf_prof = np.zeros(len(kt_prof))

                for i in range(len(kt_prof)):
                    cf_prof[i], lf_prof[i], _, _, _ = calc_emissivity(Mhyd.cosmo, Mhyd.redshift, Mhyd.nh,
                                                                      kt_prof[i], Z_ab_prof[i],
                                                                      this_sb_data.rmf, abund = Mhyd.abund, arf = this_sb_data.arf,
                                                                      elow = this_sb_data.elow, ehigh = this_sb_data.ehigh, unit = this_sb_data.unit, absorption_model = self.absorption_model, quiet = self.quiet,
                                                                      lum_elow = lum_elow, lum_ehigh = lum_ehigh)

                this_sb_data.mean_cf = np.mean(cf_prof)

                with open(out_cfact_file, 'w') as f:
                    val = [float(x) for x in np.exp(fpars) * pars_temp]
                    print(f'# V06 kT fit: {val}', file = f)
                    print('#           R           e_R            kT             Z            CR           LUM ', file = f)
                    for i in range(len(kt_prof)):
                        print(f'{this_sb_data.rref_x_am[i]:13.7g} {this_sb_data.err_rref_x_am[i]:13.7g} {kt_prof[i]:13.7g} {Z_ab_prof[i]:13.7g} {cf_prof[i]:13.7g} {lf_prof[i]:13.7g}', file = f)

                fig, axs = pl.subplots(3, 1, figsize = (14, 10), sharex = True)

                ax = axs[0]

                ax.plot(this_sb_data.rref_x_am, kt_prof, label = 'ktprof for cf')

                for this_spec_data in Mhyd.spec_data:
                    ax.errorbar(this_spec_data.rref_x_am, this_spec_data.temp_x, yerr = [this_spec_data.templ, this_spec_data.temph], fmt = 'o', ls = '--', label = 'Data', color = 'r')

                ax.set_ylabel('kT [keV]', fontsize = 18)

                ax.legend(fontsize = 18)

                ax = axs[1]

                ax.plot(this_sb_data.rref_x_am, Z_ab_prof, label = 'zfeprof for cf')

                for this_spec_data in Mhyd.spec_data:
                    ax.errorbar(this_spec_data.rref_x_am, this_spec_data.zfe, yerr = [this_spec_data.zfe_lo, this_spec_data.zfe_hi], fmt = 'o', ls = '--', label = 'Data clipped when error > 30%',
                            color = 'r')

                ax.set_ylabel('Metallicity [$Z_\\odot$]', fontsize = 18)

                ax.set_ylim([0, 1])

                ax.legend(fontsize = 18)

                ax = axs[2]

                ax.plot(this_sb_data.rref_x_am, cf_prof, label = 'LAMBDA factor', ls = '--', marker = 'o')

                ax.set_ylabel('LAMBDA MOS2', fontsize = 18)

                ax.set_xlabel('Radius [arcmin]', fontsize = 18)

                ax.legend(fontsize = 18)

                ax.set_xlim(left = 1 / 10)
                ax.set_xscale('log')

                fig.tight_layout()

                fig.subplots_adjust(hspace = 0)

                fig.savefig(out_cfact_pdf)
            else:
                rref_am, cf_prof, lf_prof = np.genfromtxt(out_cfact_file, unpack = True, usecols = (0, 4, 5))

                assert np.all(np.isclose(this_sb_data.rref_x_am, rref_am, rtol = 1e-5))

            this_sb_data.ccf, this_sb_data.llf = (cf_prof, lf_prof)

    def conversion_factor_table(self, Mhyd, lum_elow = 0.5, lum_ehigh = 2.0, log_kt_min=None, log_kt_max=None, n_kt=None, Z_ab_min=None, Z_ab_max=None, n_Z_ab=None):

        kt_array = jnp.exp(jnp.linspace(log_kt_min, log_kt_max, n_kt))
        Z_ab_array = jnp.linspace(Z_ab_min, Z_ab_max, n_Z_ab)

        for i_sb, this_sb_data in enumerate(Mhyd.sb_data):

            if not os.path.isfile(f'{Mhyd.dir}/cf_lf_table_{i_sb}.npz'):

                self.logger.info('Creating conversion factor table')

                cf_mat = jnp.zeros((len(kt_array), len(Z_ab_array)))
                lf_mat = jnp.zeros((len(kt_array), len(Z_ab_array)))

                for i, kt in tqdm(enumerate(kt_array)):
                    for j, Z_ab in enumerate(Z_ab_array):
                        cf, lf, _, _, _ = calc_emissivity(Mhyd.cosmo, Mhyd.redshift, Mhyd.nh,
                                                          kt, Z_ab,
                                                          this_sb_data.rmf, abund = Mhyd.abund, arf = this_sb_data.arf,
                                                          elow = this_sb_data.elow, ehigh = this_sb_data.ehigh, unit = this_sb_data.unit, absorption_model = self.absorption_model, quiet = self.quiet,
                                                          lum_elow = lum_elow, lum_ehigh = lum_ehigh)
                        cf_mat = cf_mat.at[i, j].set(cf)
                        lf_mat = lf_mat.at[i, j].set(lf)


                jnp.savez(f'{Mhyd.dir}/cf_lf_table_{i_sb}', cf_mat = cf_mat, lf_mat = lf_mat, kt_array = kt_array, Z_ab_array = Z_ab_array)

            else:
                with jnp.load(f'{Mhyd.dir}/cf_lf_table_{i_sb}.npz') as f:
                    content = dict(f)
                kt_array_saved = content['kt_array']
                Z_ab_array_saved = content['Z_ab_array']
                assert jnp.all(jnp.isclose(kt_array_saved, kt_array)) and jnp.all(np.isclose(Z_ab_array_saved, Z_ab_array)), (f'Content of {Mhyd.dir}/cf_lf_table.npz does not match expected input arrays. '
                                                                                                                              f'Saved: {kt_array_saved[0]},{kt_array_saved[-1]},{len(kt_array_saved)} and '
                                                                                                                              f'{Z_ab_array_saved[0]},{Z_ab_array_saved[-1]},{len(Z_ab_array_saved)}. '
                                                                                                                              f'Expected: {kt_array[0]},{kt_array[-1]},{len(kt_array)} and '
                                                                                                                              f'{Z_ab_array[0]},{Z_ab_array[-1]},{len(Z_ab_array)}. ')
                cf_mat = jnp.array(content['cf_mat'])
                lf_mat = jnp.array(content['lf_mat'])

            this_sb_data.cf_mat = cf_mat
            this_sb_data.lf_mat = lf_mat
            this_sb_data.kt_array = kt_array
            this_sb_data.Z_ab_array = Z_ab_array

def build_design_matrix_2d(x, y, degree_x, degree_y):
    """
    Construct polynomial design matrix for total degree <= degree.

    Assumes x and y are already scaled appropriately.
    """
    powers = [
        (i, j)
        for i, j in product(range(max(degree_x, degree_y) + 1), repeat = 2)
        if i <= degree_x and j <= degree_y
    ]

    A = jnp.column_stack([
        (x.ravel() ** i) * (y.ravel() ** j)
        for i, j in powers
    ])

    return A, powers

def poly2d_model(A, f_obs = None):
    """
    Bayesian 2D polynomial regression model.
    """
    n_params = A.shape[1]

    # Hierarchical shrinkage prior for regularization
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))

    coeffs_num = numpyro.sample(
        "coeffs_num",
        dist.Normal(0.0, tau).expand([n_params])
    )

    coeffs_den = numpyro.sample(
        "coeffs_den",
        dist.Normal(0.0, tau).expand([n_params])
    )

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    num = jnp.dot(A, coeffs_num)
    den = jnp.dot(A, coeffs_den)

    mu = num / den

    numpyro.sample(
        "obs",
        dist.Normal(mu, sigma),
        obs = f_obs
    )

def fit_poly2d_numpyro(
        x,
        y,
        f,
        degree_x = 3,
        degree_y = 3,
        num_warmup = 1000,
        num_samples = 2000,
        rng_seed = 0,
):
    """
    Fit Bayesian 2D polynomial using NumPyro.

    Parameters
    ----------
    x, y : ndarray
        Scaled coordinate arrays (same shape as f)
    f : ndarray
        Function values
    degree : int
        Maximum total polynomial degree

    Returns
    -------
    samples : dict
        Posterior samples
    powers : list
        Polynomial powers corresponding to coefficients
    A : ndarray
        Design matrix
    """
    A, powers = build_design_matrix_2d(x, y, degree_x, degree_y)

    kernel = NUTS(poly2d_model, target_accept_prob = 0.95, dense_mass = True)

    mcmc = MCMC(
        kernel,
        num_warmup = num_warmup,
        num_samples = num_samples,
    )

    rng_key = jrandom.PRNGKey(rng_seed)

    mcmc.run(
        rng_key,
        A = A,
        f_obs = f.ravel(),
    )

    samples = mcmc.get_samples()

    return samples, powers, A

@attrs.define(kw_only = True, slots = False)
class FitCCFTable:
    logger: Logger = attrs.field(validator = attrs.validators.instance_of(Logger), init = False)

    cf_mat: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), repr = False, init = True)
    lf_mat: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), repr = False, init = True)

    log_kt_array: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), repr = False, init = True)
    Z_ab_array: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), repr = False, init = True)

    log_cf: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), repr = False, init = False)
    log_lf: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), repr = False, init = False)

    log_cf_min: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    log_cf_max: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    log_lf_min: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    log_lf_max: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    log_kt_min: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    log_kt_max: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    Z_ab_min: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)
    Z_ab_max: float = attrs.field(validator = attrs.validators.instance_of(float), init = False)

    powers: list = attrs.field(validator = attrs.validators.instance_of(list), init = False)

    coeff_num_mean: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), init = False)
    coeff_num_std: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), init = False)

    coeff_den_mean: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), init = False)
    coeff_den_std: jnp.ndarray = attrs.field(validator = attrs.validators.instance_of(jnp.ndarray), init = False)

    coeff_num_mean_np: np.ndarray = attrs.field(validator = attrs.validators.instance_of(np.ndarray), init = False)
    coeff_den_mean_np: np.ndarray = attrs.field(validator = attrs.validators.instance_of(np.ndarray), init = False)

    directory: str = attrs.field(validator = attrs.validators.instance_of(str), init = False)
    i_sb: int = attrs.field(validator = attrs.validators.instance_of(int), init = False)

    def __attrs_post_init__(self):
        self.logger = create_logger('FitCCFTable')


        self.log_cf = jnp.log(self.cf_mat)
        self.log_lf = jnp.log(self.lf_mat)

        self.log_cf_min = float(jnp.min(self.log_cf))
        self.log_cf_max = float(jnp.max(self.log_cf))

        self.log_lf_min = float(jnp.min(self.log_lf))
        self.log_lf_max = float(jnp.max(self.log_lf))

        self.log_kt_min = float(jnp.min(self.log_kt_array))
        self.log_kt_max = float(jnp.max(self.log_kt_array))

        self.Z_ab_min = float(jnp.min(self.Z_ab_array))
        self.Z_ab_max = float(jnp.max(self.Z_ab_array))


    def renorm_log_kt(self, log_kt):
        return 2 * (log_kt - self.log_kt_min) / (self.log_kt_max - self.log_kt_min) - 1

    def renorm_Z_ab(self, Z_ab):
        return 2 * (Z_ab - self.Z_ab_min) / (self.Z_ab_max - self.Z_ab_min) - 1

    def renorm_log_cf(self, log_cf):
        return 2 * (log_cf - self.log_cf_min) / (self.log_cf_max - self.log_cf_min) - 1

    def renorm_log_lf(self, log_lf):
        return 2 * (log_lf - self.log_lf_min) / (self.log_lf_max - self.log_lf_min) - 1


    def fit_cf(self, directory, i_sb):

        fitted = False

        if not os.path.isfile(f'{directory}/cf_lf_fit_{i_sb}.npz'):

            fitted = True

            self.logger.info('Fitting conversion factor table')

            x_renorm = self.renorm_log_kt(self.log_kt_array)

            y_renorm = self.renorm_Z_ab(self.Z_ab_array)

            cf_renorm = self.renorm_log_cf(self.log_cf)

            lf_renorm = self.renorm_log_lf(self.log_lf)

            Y, X = jnp.meshgrid(y_renorm, x_renorm)

            samples_cf, powers_cf, A = fit_poly2d_numpyro(
                X,
                Y,
                cf_renorm,
                degree_x = 21,
                degree_y = 5,
            )

            coeff_num_mean_cf = samples_cf["coeffs_num"].mean(axis = 0)
            coeff_num_std_cf = samples_cf["coeffs_num"].std(axis = 0)
            coeff_den_mean_cf = samples_cf["coeffs_den"].mean(axis = 0)
            coeff_den_std_cf = samples_cf["coeffs_den"].std(axis = 0)

            content = dict()
            content['powers_cf'] = powers_cf
            content['coeff_num_mean_cf'] = coeff_num_mean_cf
            content['coeff_num_std_cf'] = coeff_num_std_cf
            content['coeff_den_mean_cf'] = coeff_den_mean_cf
            content['coeff_den_std_cf'] = coeff_den_std_cf

            samples_lf, powers_lf, A = fit_poly2d_numpyro(
                X,
                Y,
                lf_renorm,
                degree_x=21,
                degree_y=5,
            )

            coeff_num_mean_lf = samples_lf["coeffs_num"].mean(axis=0)
            coeff_num_std_lf = samples_lf["coeffs_num"].std(axis=0)
            coeff_den_mean_lf = samples_lf["coeffs_den"].mean(axis=0)
            coeff_den_std_lf = samples_lf["coeffs_den"].std(axis=0)

            content['powers_lf'] = powers_lf
            content['coeff_num_mean_lf'] = coeff_num_mean_lf
            content['coeff_num_std_lf'] = coeff_num_std_lf
            content['coeff_den_mean_lf'] = coeff_den_mean_lf
            content['coeff_den_std_lf'] = coeff_den_std_lf

            jnp.savez(f'{directory}/cf_lf_fit_{i_sb}', **content)

        else:

            self.logger.info(f'Fit already performed. Loading from file {directory}/cf_lf_{i_sb}.npz')

            with jnp.load(f'{directory}/cf_lf_fit_{i_sb}.npz') as f:
                content = dict(f)

            powers_cf = list(content['powers_cf'])
            coeff_num_mean_cf = jnp.array(content['coeff_num_mean_cf'], dtype = 'float64')
            coeff_num_std_cf = jnp.array(content['coeff_num_std_cf'], dtype = 'float64')
            coeff_den_mean_cf = jnp.array(content['coeff_den_mean_cf'], dtype = 'float64')
            coeff_den_std_cf = jnp.array(content['coeff_den_std_cf'], dtype = 'float64')

            powers_lf = list(content['powers_lf'])
            coeff_num_mean_lf = jnp.array(content['coeff_num_mean_lf'], dtype='float64')
            coeff_num_std_lf = jnp.array(content['coeff_num_std_lf'], dtype='float64')
            coeff_den_mean_lf = jnp.array(content['coeff_den_mean_lf'], dtype='float64')
            coeff_den_std_lf = jnp.array(content['coeff_den_std_lf'], dtype='float64')

        self.coeff_num_mean_cf = coeff_num_mean_cf
        self.coeff_den_mean_cf = coeff_den_mean_cf
        self.coeff_num_mean_cf_np = np.asarray(coeff_num_mean_cf)
        self.coeff_den_mean_cf_np = np.asarray(coeff_den_mean_cf)
        self.powers_cf = powers_cf

        self.coeff_num_mean_lf = coeff_num_mean_lf
        self.coeff_den_mean_lf = coeff_den_mean_lf
        self.coeff_num_mean_lf_np = np.asarray(coeff_num_mean_lf)
        self.powers_lf = powers_lf

        if fitted:
            self.show_fit(directory, i_sb)

    def show_fit(self, directory, i_sb):

        x_renorm = self.renorm_log_kt(self.log_kt_array)

        y_renorm = self.renorm_Z_ab(self.Z_ab_array)

        f_renorm = self.renorm_log_cf(self.log_cf)

        Y, X = jnp.meshgrid(y_renorm, x_renorm)

        cf_mean = self.predict_poly2d_mean_cf(X, Y)
        lf_mean = self.predict_poly2d_mean_lf(X, Y)

        fig, axes = pl.subplots(1, 3, figsize = (18, 5))

        # Original data
        im0 = axes[0].imshow(
            f_renorm,
            origin = "lower",
            aspect = "auto"
        )
        axes[0].set_title("Original cf")
        pl.colorbar(im0, ax = axes[0])

        # Polynomial fit
        im1 = axes[1].imshow(
            cf_mean,
            origin = "lower",
            aspect = "auto"
        )
        axes[1].set_title("Polynomial Fit")
        pl.colorbar(im1, ax = axes[1])

        # Residuals
        im2 = axes[2].imshow(
            f_renorm - cf_mean,
            origin = "lower",
            aspect = "auto"
        )
        axes[2].set_title("Residuals")
        pl.colorbar(im2, ax = axes[2])

        pl.tight_layout()
        pl.savefig(f'{directory}/cf_residuals_{i_sb}.pdf')
        pl.close()

        cf_best_fit = np.exp((cf_mean + 1) * (self.log_cf_max - self.log_cf_min) / 2 + self.log_cf_min)
        lf_best_fit = np.exp((lf_mean + 1) * (self.log_lf_max - self.log_lf_min) / 2 + self.log_lf_min)

        fig, axs = pl.subplots(2, 1, figsize = (14, 10), sharex = True, height_ratios = [3, 1])

        ax = axs[0]
        for i, tZ in enumerate(self.Z_ab_array):
            ax.loglog(np.exp(self.log_kt_array), self.cf_mat[:, i], ls = ':', lw = 0.1, marker = 'o', label = f'True Z={tZ:.2f}', color=f'C{i}')
            ax.loglog(np.exp(self.log_kt_array), cf_best_fit[:, i], ls = '-', label = f'Model Z={tZ:.2f}', color=f'C{i}')
        ax.set_ylabel('Conversion factor [cts/s]')

        # ax.loglog(kt_array, cf_best_fit2)
        ax.legend()

        ax = axs[1]
        ax.plot(np.exp(self.log_kt_array), (cf_best_fit - self.cf_mat) / self.cf_mat)
        ax.axhline(0, ls = '--', color = 'k')
        # ax.plot(kt_array, (cf_mat-cf_best_fit2)/cf_mat, ls=':')
        # pl.xscale('log')
        ax.set_ylabel('(Fit-True)/Fit')
        ax.set_xlabel('Temperature [keV]')
        ax.set_ylim(-0.005, 0.005)
        fig.tight_layout()

        fig.subplots_adjust(hspace = 0)

        fig.savefig(f'{directory}/cf_fit_{i_sb}.pdf')
        pl.close()

        fig, axs = pl.subplots(2, 1, figsize = (14, 10), sharex = True, height_ratios = [3, 1])

        ax = axs[0]
        for i, tZ in enumerate(self.Z_ab_array):
            ax.loglog(np.exp(self.log_kt_array), self.lf_mat[:, i], ls = ':', lw = 0.1, marker = 'o', label = f'True Z={tZ:.2f}', color=f'C{i}')
            ax.loglog(np.exp(self.log_kt_array), lf_best_fit[:, i], ls = '-', label = f'Model Z={tZ:.2f}', color=f'C{i}')
        ax.set_ylabel('Conversion factor [cts/s]')

        ax.legend()

        ax = axs[1]
        ax.plot(np.exp(self.log_kt_array), (lf_best_fit - self.lf_mat) / self.lf_mat)
        ax.axhline(0, ls = '--', color = 'k')
        # pl.xscale('log')
        ax.set_ylabel('(Fit-True)/Fit')
        ax.set_xlabel('Temperature [keV]')
        ax.set_ylim(-0.005, 0.005)
        fig.tight_layout()

        fig.subplots_adjust(hspace = 0)

        fig.savefig(f'{directory}/lf_fit_{i_sb}.pdf')
        pl.close()

    def predict_poly2d_mean_cf(self, x_new, y_new):
        """
        Posterior predictive mean/std at arbitrary points.
        """

        A_new = jnp.vstack([(x_new.ravel() ** i) * (y_new.ravel() ** j) for i, j in self.powers_cf])

        num = self.coeff_num_mean_cf @ A_new
        den = self.coeff_den_mean_cf @ A_new

        pred = num / den

        return pred.reshape(x_new.shape)

    def predict_poly2d_mean_lf(self, x_new, y_new):
        """
        Posterior predictive mean/std at arbitrary points.
        """

        A_new = jnp.vstack([(x_new.ravel() ** i) * (y_new.ravel() ** j) for i, j in self.powers_lf])

        num = self.coeff_num_mean_lf @ A_new
        den = self.coeff_den_mean_lf @ A_new

        pred = num / den

        return pred.reshape(x_new.shape)

    def predict_cf(self, log_kt, Z):

        log_kt = np.atleast_1d(log_kt)
        Z = np.atleast_1d(Z)

        x_new = self.renorm_log_kt(log_kt)
        y_new = self.renorm_Z_ab(Z)

        A_new = jnp.vstack([(x_new.ravel() ** i) * (y_new.ravel() ** j) for i, j in self.powers_cf])

        num = self.coeff_num_mean_cf @ A_new
        den = self.coeff_den_mean_cf @ A_new

        pred = num / den

        cf = np.exp((pred + 1) * (self.log_cf_max - self.log_cf_min) / 2 + self.log_cf_min)

        return cf.reshape(log_kt.shape)

    def predict_lf(self, log_kt, Z):

        log_kt = np.atleast_1d(log_kt)
        Z = np.atleast_1d(Z)

        x_new = self.renorm_log_kt(log_kt)
        y_new = self.renorm_Z_ab(Z)

        A_new = jnp.vstack([(x_new.ravel() ** i) * (y_new.ravel() ** j) for i, j in self.powers_lf])

        num = self.coeff_num_mean_lf @ A_new
        den = self.coeff_den_mean_lf @ A_new

        pred = num / den

        cf = np.exp((pred + 1) * (self.log_lf_max - self.log_lf_min) / 2 + self.log_lf_min)

        return cf.reshape(log_kt.shape)


    def predict_cf_pm(self, log_kt, Z):

        x_new = self.renorm_log_kt(log_kt)
        y_new = self.renorm_Z_ab(Z)

        # 1. Recreate the Vandermonde-style matrix (A_new)
        # We use pt.stack instead of jnp.vstack
        A_list = [(x_new ** i) * (y_new ** j) for i, j in self.powers_cf]
        A_new = tt.stack(A_list, axis = 0)
        #

        # Now the @ operator will work because PyTensor knows
        # how to wrap NumPy arrays into the graph.
        num = self.coeff_num_mean_cf_np @ A_new
        den = self.coeff_den_mean_cf_np @ A_new

        # 3. Element-wise division and transformation
        pred = num / den

        # 4. Final exponential transformation
        # Use pt.exp instead of np.exp or jnp.exp
        cf = tt.exp((pred + 1) * (self.log_cf_max - self.log_cf_min) / 2 + self.log_cf_min)

        return cf
