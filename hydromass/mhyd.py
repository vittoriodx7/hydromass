import os
import pprint

import ipdb

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import logging
import sys

import numpy as np

isjax = False
try:
    import jax
    import numpyro
except ImportError:
    print('JAX/numpyro not found, using default sampler')
else:
    print('We will use the JAX sampler')
    isjax = True

import pymc as pm
import arviz as az

from .constants import cgsamu, cgskpc, Msun, y_prefactor, kev2erg, cgsMpc, cgsG
from .pnt import get_data_file_path, alpha_turb_pm, NPmodel
from .mean_molecular_weights import mean_molecular_weights
from .pyutils import create_logger, warning
from .conversion_factor_emissivity import ConversionFactor, FitCCFTable
from .utility import sb_utils, common_utils, spec_utils, mass_utils
from .likelihoods import get_density, likelihood_sb, get_metal, likelihood_metal, kt_from_mass, kt_from_forward, likelihood_kt, likelihood_sz, get_ccf
from .show_fit import compute_products, _corner_inputs, save_products, plot_likelihood, plot_mass, plot_mass_fractions, plot_corner, plot_thermodynamics
from .get_delta import calc_delta_quantities
from .model import Model
from .model_forward import ForwardModel


__all__ = ['Mhyd']


class Mhyd:
    """

    The Mhyd class is the core class of hydromass. It allows the user to pass one or more datasets to be fitted, chose the mass reconstruction method, set options like change cosmology, Solar abundance table, NUTS options, model choice, and more.

    :param sbprofile: A pyproffit Profile object (https://pyproffit.readthedocs.io/en/latest/pyproffit.html#module-pyproffit.profextract) including the surface brightness data
    :type sbprofile: class:`pyproffit.profextract.Profile`
    :param spec_data: A :class:`hydromass.tpdata.SpecData` object including a spectroscopic X-ray temperature profile and its associated uncertainties
    :type spec_data: class:`hydromass.tpdata.SpecData`
    :param sz_data: A :class:`hydromass.tpdata.SZData` object containing an SZ pressure profile and its covariance matrix
    :type sz_data: class:`hydromass.tpdata.SZData`
    :param directory: Name of output file directory. Defaults to 'mhyd'
    :type directory: str
    :param redshift: Source redshift
    :type redshift: float
    :param cosmo: Astropy cosmological model
    :type cosmo: class:`astropy.cosmology`
    :param abund: Solar abundance table. Available are 'angr' (Anders & Grevesse 1987), 'aspl' (Asplund et al. 2009), and 'grsa' (Grevesse & Sauval 2005). Defaults to 'aspl'
    :type abund: str
    :param max_rad: Maximum radius out where cluster model is considered. Default 0 kpc to go back to the data
    :type max_rad: float
    """

    def __init__(self,
                 abund = 'aspl', Zs = 0.3,
                 galaxy_object = None, directory = None,
                 sb_data = None, spec_data = None, sz_data = None, wl_data = None, vel_data = None,
                 cosmo = None,
                 max_rad = 0.,
                 tag = None
                 ):

        logger = logging.getLogger('hydromass')
        if not logger.hasHandlers():
            # Fallback: Define the "default" here because nothing else exists
            logger.setLevel(logging.INFO)
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(logging.Formatter('%(asctime)s @%(name)-25s "%(message)s"', datefmt = '%d-%m-%y %H:%M'))
            logger.addHandler(sh)

        self.logger = create_logger()

        if tag is not None:
            self.logger.info(f'This run tag is {tag}')
        self.tag = tag

        self.abund, self.mup, self.nhc, self.mu_e = mean_molecular_weights(abund = abund, Zs = Zs)

        if directory is None:
            directory = f'mhyd_tmp{random.randint(0, 1000)}'  # random number
            self.logger.info(f'No output directory provided, will output to subdirectory "{directory}" ')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.logger.info(f'Output folder is {directory}')
        self.dir = directory

        if galaxy_object is None:
            raise ValueError('No galaxy_object (type hydromass.GalaxyObject provided, please provide one with the "galaxy_object=" option')
        self.redshift = galaxy_object.redshift
        self.nh = galaxy_object.nh

        if sb_data is None:
            raise ValueError('no surface brightness profile provided, please provide one with the "sb_data=" option')
        if type(sb_data) is not list:
            logger.info('sb_data is not a list. Converting it... For future version sb_data will always be treated as a list')
            sb_data = [sb_data]
        self.sb_data = sb_data




        if cosmo is None:
            self.logger.info('No cosmology provided, will default to Planck15')
            from astropy.cosmology import Planck15 as cosmo
        self.cosmo = cosmo

        self.dlum = float(cosmo.luminosity_distance(self.redshift).value)
        self.logger.info(f'Luminosity distance to the source: {self.dlum:.4g} Mpc')

        self.amin2kpc = float(cosmo.kpc_proper_per_arcmin(self.redshift).value)
        self.logger.info(f'At the redshift of the source 1 arcmin is {self.amin2kpc:.4g} kpc')

        self.rho_cz = (cosmo.critical_density(self.redshift).to('Msun/Mpc**3')).value # * cgsMpc ** 3 / Msun # critical density in Msun per Mpc^3
        self.logger.info(f'At the redshift of the source critical density is {self.rho_cz:.4g} Msun/Mpc^3')

        if spec_data is None and sz_data is None:
            raise ValueError('Error: no spectral data file or SZ data file provided, please provide at least one with the "spec_data=" or "sz_data=" options')
        if type(spec_data) is not list:
            logger.info('spec_data is not a list. Converting it... For future version spec_data will always be treated as a list')
            spec_data = [spec_data]
        if type(sz_data) is not list:
            logger.info('sz_data is not a list. Converting it... For future version sz_data will always be treated as a list')
            sz_data = [sz_data]

        if all(v is None for v in spec_data):
            spec_data = None
        if all(v is None for v in sz_data):
            sz_data = None

        self.spec_data = spec_data
        self.sz_data = sz_data

        # Make sure the conversion from arcmin to kpc is done in the correct cosmology, i.e. use amin2kpc just calculated
        for sb in self.sb_data:
            sb.kpc_data(self.amin2kpc)
        if self.spec_data is not None:
            for sp in self.spec_data:
                if sp is not None:
                    sp.kpc_data(self.amin2kpc)
        if self.sz_data is not None:
            for sz in self.sz_data:
                if sz is not None:
                    sz.kpc_data(self.amin2kpc)

        # TODO: Do the same for WL and sigma_V
        self.wl_data = wl_data
        self.veldata = vel_data

        self.mfact = 4. * np.pi * self.rho_cz * 1e-22
        self.mfact0 = kev2erg * cgskpc / (cgsG * cgsamu * self.mup) / Msun / 1e13
        self.mgas_fact = cgsamu * self.mu_e / Msun
        self.transf = 4. * (1. + self.redshift) ** 2 * (180. * 60.) ** 2 / np.pi / 1e-14 * self.nhc / cgsMpc * 1e3


        rmax_sb, rmax_kt, rmax_sz = 0, 0, 0
        if self.sb_data is not None:
            for this_sb_data in self.sb_data:
                this_rmax_sb = np.max(this_sb_data.rout_x_kpc)
                if this_rmax_sb > rmax_sb:
                    rmax_sb = this_rmax_sb
        if self.spec_data is not None:
            for this_spec_data in self.spec_data:
                this_rmax_kt = np.max(this_spec_data.rout_x_kpc)
                if this_rmax_kt > rmax_kt:
                    rmax_kt = this_rmax_kt
        if self.sz_data is not None:
            for this_sz_data in self.sz_data:
                if this_sz_data is not None:
                    this_rmax_sz = np.max(this_sz_data.rout_sz_kpc)
                    if this_rmax_sz > rmax_sz:
                        rmax_sz = this_rmax_sz

        if max_rad < max(rmax_sb, rmax_kt, rmax_sz):
            max_rad = max(rmax_sb, rmax_kt, rmax_sz)
            self.logger.info(f'WARNING: changing maximum radius of integration to {max_rad}')

        self.max_rad_kpc = max_rad
        self.max_rad_am = max_rad / self.amin2kpc

    def emissivity(self, absorption_model='phabs', conversion_factor_type='table', quiet=True, lum_elow=0.5, lum_ehigh=2.0):
        '''
        In the updated version of hydromass, [rmf, arf, elow, ehigh, unit] are properties of the sb_data, while [redshift, nh] are properties of Mhyd from GalaxyObject

        :param absorption_model:
            e.g. phabs, tbabs, etc. used to absorbed photons from the source to the observer
        :param conversion_factor_type:
            If type='variable', the temperature and metallicity profile will be fitted. If type='single', average temperature and metallicity are used. If type='table' a table in a grid of temperature
            and metallicity is computed; then luminosity and count rate conversion factor are fitted using a polynomial ratio
        :param quiet:
            Controls the verbosity of Xspec
        :param lum_elow:
            Low energy bound (rest frame) for luminosity calculation.
        :param lum_ehigh:
            High energy bound (rest frame) for luminosity calculation.
        '''
        ConversionFactor.from_mhyd(
            self, absorption_model, quiet, conversion_factor_type, lum_elow=lum_elow, lum_ehigh=lum_ehigh
        )
        self.conversion_factor_type = conversion_factor_type

        if conversion_factor_type == 'table':
            import jax.numpy as jnp

            # for each sb_data fit table and return a class
            for i_sb, this_sb_data in enumerate(self.sb_data):
                fit = FitCCFTable(cf_mat=this_sb_data.cf_mat,
                                  lf_mat=this_sb_data.lf_mat,
                                  log_kt_array=jnp.log(this_sb_data.kt_array),
                                  Z_ab_array=this_sb_data.Z_ab_array
                    )
                fit.fit_cf(self.dir, i_sb)
                this_sb_data.cf_fit = fit

                kt = float(np.mean([np.average(this_spec_data.temp_x, weights=1. / this_spec_data.errt_x ** 2) for this_spec_data in self.spec_data]))
                means = []
                for d in self.spec_data:
                    sel = d.errt_zfe > 0
                    if not sel.any():
                        continue
                    means.append(np.average(d.zfe[sel], weights=1. / d.errt_zfe[sel] ** 2))
                Z_ab = float(np.mean(means))
                this_sb_data.mean_cf = fit.predict_cf(np.log(kt), Z_ab)
                this_sb_data.mean_lf = fit.predict_lf(np.log(kt), Z_ab)

                # If variable_ccf was calculated, compare the resulting ccf
                variable_ccf = f'{self.dir}/emissivity_{i_sb}.dat'
                if os.path.isfile(variable_ccf):
                    rref_am, kt_prof, Z_prof, cf_prof, lf_prof = np.genfromtxt(variable_ccf, unpack=True, usecols=(0, 2, 3, 4, 5))

                    cf_predict = fit.predict_cf(np.log(kt_prof), Z_prof)
                    lf_predict = fit.predict_lf(np.log(kt_prof), Z_prof)

                    # print(cf_prof / cf_predict)

                    import matplotlib.pyplot as pl
                    fig, ax1 = pl.subplots(figsize=(8, 5))

                    ax1.plot(rref_am, cf_prof, label='Variable ccf', color='b')
                    ax1.plot(rref_am, cf_predict, label='Table ccf', color='b', ls='--')
                    ax1.tick_params(axis='y', labelcolor='b')

                    ax2 = ax1.twinx()
                    ax2.plot(rref_am, lf_prof, label='Variable llf', color='r')
                    ax2.plot(rref_am, lf_predict, label='Table llf', color='r', ls='--')
                    ax2.tick_params(axis='y', labelcolor='r')

                    ax1.set_xscale('log')
                    ax1.legend()
                    ax2.legend()
                    pl.savefig(f'{self.dir}/cf_table_variable_comparison.pdf')
                    pl.close()


    def store_fit_setup(self, fit_args):
        '''
        CLAUDE: Store the arguments of :meth:`fit` as a dictionary in self.fit_dir/setup.log.
        Models are stored as a dictionary with their class name and attributes (functions excluded), numpy arrays as lists.

        :param fit_args: Dictionary of the arguments passed to :meth:`fit`
        '''
        def to_setup(value):
            if isinstance(value, Model):
                return {'class': type(value).__name__, **{k: to_setup(v) for k, v in vars(value).items() if not callable(v)}}
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return type(value)(to_setup(v) for v in value)
            if isinstance(value, dict):
                return {k: to_setup(v) for k, v in value.items()}
            return value

        setup_path = os.path.join(self.fit_dir, 'setup.log')
        with open(setup_path, 'w') as f:
            f.write(pprint.pformat(to_setup(fit_args), sort_dicts=False) + '\n')
        self.logger.info(f'Fit setup saved to {setup_path}')

    def quick_fit_density(self, dict_utils, fit_bkg=False):
        '''
        CLAUDE: Quick maximum a posteriori fit of the surface brightness profiles with the density model only and a single
        conversion factor per profile (sb_data.mean_cf, set by :meth:`emissivity` for any conversion_factor_type).
        The best-fit coefs (and bkg if fit_bkg) are stored in dict_utils['coefs_start'] (and dict_utils['bkg_start']),
        used by :func:`hydromass.likelihoods.get_density` as starting values.

        :param dict_utils: Dictionary filled by :func:`hydromass.utility.common_utils` and :func:`hydromass.utility.sb_utils`
        :param fit_bkg: Whether BKG is to be fitted or not. Default False.
        '''
        self.logger.info('Running quick density fit with a single conversion factor...')
        with pm.Model():
            params = dict()

            get_density(params, dict_utils, self, fit_bkg=fit_bkg)

            for i, this_sb_data in enumerate(self.sb_data):
                params[f'ccf_{i}'] = this_sb_data.mean_cf

            likelihood_sb(params, dict_utils, self, fit_bkg=fit_bkg)

            start = pm.find_MAP(method='L-BFGS-B', maxeval=5000)

        dict_utils['coefs_start'] = start['coefs']
        if fit_bkg:
            dict_utils['bkg_start'] = start['bkg']

        self.logger.info(f"Found initial values for density: {dict_utils['coefs_start']}")

    def fit_massmod(self,
            fit_SB=False, fit_bkg=False, bkglim_amin=None, backval=None, quick_density=True,
            fit_Z_Ab=False, model_Z_Ab=None,
            fit_kT=False,
            fit_SZ=False,
            fit_WL=False,
            fit_eta=False,
            fit_elong=False,
            cM_relation=False,
            models_mass=None,
            model_mstar=None, mstar=None,
            model_missing_mass=None,
            sb_setup=None,
            rmin_amin=None, rmax_amin=None,
            p0_prior=None, p0_type='sb',
            num_warmup=1000, num_samples=2000, target_accept=0.95, num_chains=4, nuts_sampler="numpyro",
            find_map=True, deltas=(2500, 1000, 500, 200), auto_plot=True, use_clip=True,
            nmore = 5
            ):
        '''
        :param fit_SB:
            Whether to fit the surface brightness profile or not. Default False.
        :param fit_bkg:
            Whether BKG is to be fitted or not. Default False: BKG subtracted and error added in quadrature to SB points.
        :param bkglim_amin:
            Region beyond which estimate the background value. Default last data point.
        :param backval:
            Estimated background value. Default None: compute from bkglim_amin parameter.
        :param quick_density:
            Whether to run :meth:`quick_fit_density` before the fit, to use its density coefficients (and BKG) as starting values. Default True.
        :param fit_Z_Ab:
            Whether to fit the metallicity profile. Default False.
        :param model_Z_Ab:
            Model used to fit the metallicity profile. Default None.
        :param fit_kT:
            Whether to fit the temperature profile. Default False.
        :param fit_SZ:
            Whether to fit the SZ profile. Default False.
        :param fit_WL:
            Whether to fit the WL profile. Default False.
        :param fit_eta:
            Whether to allow the ratio between X-ray and SZ pressure to be a free parameter. Default False.
        :param fit_elong:
        :param cM_relation:
        :param models_mass:
            List of models to fit the non baryonic mass component.
            The total mass is assumed to be: Mtot = Mgas + Mstar + Mother, with Mgas given by the gas mass from the integral of the gas density,
            Mstar given by star mass component, and Mother given by the sum of models_mass
        :param mstar:
            Array containing the cumulative stellar mass profile.
        :param model_mstar:
            Model used to fit the stellar mass component.
                if mstar=None and model_mstar=None: mass_star=0
                if mstar=something and model_mstar=None: mass_star=mstar
                if mstar=None and model_mstar=something: mass_star=model_mstar
                if mstar=something and model_mstar=something: mass_star=model_mstar ; model_mstar fitted to mstar
        :param model_missing_mass:
            Missing baryonic mass model: mbar = mgas + mstar + m_missing_mass_model. Default=None
        :param sb_setup:
        :param rmin_amin:
            Minimun radius to consider the profiles for the fitting. Default 0.
        :param rmax_amin:
            Maximum radius to consider the profiles for the fitting. Default max_rad_am
        :param p0_prior:
        :param p0_type:
        :param num_warmup:
            Number of steps to do in warmup (discarded for posterior evaluation). Default 1000
        :param num_samples:
            Number of steps to sample the posterior. Default 2000
        :param target_accept:
            Target acceptance probability to tune the step size. We want step size to be accepted with this probability. Default 0.95
        :param num_chains:
            Number of chains to run in parallel. Default 4.
        :param nuts_sampler:
            Sampler used for the fitting. Default numpyro
        :param find_map:
            Whether to run a maximum a posteriori fit first to initialize the sampler. Default True
        :param deltas:
            Overdensities at which R_delta, M_delta, Mgas, Mstar and mass fractions are computed and saved to mdelta.dat. Default (2500, 1000, 500, 200)
        :param auto_plot:
            Whether to save the likelihood, mass, mass fractions, corner and thermodynamics plots in the fit folder at the end of the fit. Default True
        :param use_clip:
            Whether to clip the metallicity model to [0.01, 1] and the log of the projected temperature used for the conversion factor
            to the range of the conversion factor table. Default True
        '''
        # All the arguments of fit, stored in setup.log by store_fit
        fit_args = {k: v for k, v in locals().items() if k != 'self'}

        if self.tag is None:
            # Name of the mass model combination, e.g. NFW+SERSIC, used in plot names and legends
            model_tag = '+'.join([m.massmod for m in models_mass] + [m.massmod for m in (model_mstar, model_missing_mass) if m is not None])
            if fit_SB:
                model_tag += '_SB'
            if fit_kT:
                model_tag += '_kT'
            if fit_SZ:
                model_tag += '_SZ'
                if fit_eta:
                    model_tag += '_eta'
            if fit_Z_Ab:
                model_tag += '_Z'
            if self.conversion_factor_type != 'table':
                model_tag += '_fixedccf'
        else:
            model_tag = self.tag

        # All the outputs of this fit go to a folder named after the model combination
        self.fit_dir = os.path.join(self.dir, model_tag)
        os.makedirs(self.fit_dir, exist_ok=True)

        trace_path = os.path.join(self.fit_dir, f'trace_{model_tag}.nc')
        if not os.path.isfile(trace_path):
            self.store_fit_setup(fit_args)

            if (fit_SB is False) and (fit_kT is False) and (fit_Z_Ab is False) and (fit_SZ is False) and (fit_WL is False):
                raise ValueError('Nothing to do. At least one of [fit_SB, fit_kT, fit_Z_Ab, fit_SZ, fit_WL] must be set to True')

            if fit_Z_Ab and model_Z_Ab is None:
                raise ValueError('fit_Z_Ab is True but model_Z_Ab is not set')

            dict_utils = common_utils(self, rmin_amin=rmin_amin, rmax_amin=rmax_amin, nmore = nmore)

            sb_utils(self, dict_utils, sb_setup=sb_setup, fit_bkg=fit_bkg, bkglim_amin=bkglim_amin, backval=backval)

            if quick_density:
                self.quick_fit_density(dict_utils, fit_bkg=fit_bkg)

            if fit_kT:
                spec_utils(self, dict_utils)

            # ccf_utils(self, dict_utils)

            mass_utils(self, dict_utils, mstar=mstar, p0_prior=p0_prior, p0_type=p0_type)



            hydro_model = pm.Model()
            with hydro_model:
                params = dict()

                get_density(params, dict_utils, self, fit_bkg=fit_bkg)

                get_metal(params, model_Z_Ab, self, dict_utils, use_clip=use_clip)

                if fit_Z_Ab:
                    likelihood_metal(params, model_Z_Ab, self, dict_utils)

                kt_from_mass(params, models_mass, self, dict_utils, mstar=mstar, model_mstar=model_mstar, model_missing_mass=model_missing_mass,
                             use_clip=use_clip)

                if fit_kT:
                    likelihood_kt(params, self, dict_utils)

                if fit_SZ:
                    likelihood_sz(params, self, dict_utils, fit_eta=fit_eta)

                get_ccf(params, self, dict_utils, self.conversion_factor_type)


                if fit_SB:
                    likelihood_sb(params, dict_utils, self, fit_bkg=fit_bkg)


            self.logger.info('Starting the sampler...')

            if nuts_sampler in ('numpyro', 'blackjax') and not isjax:
                self.logger.info(f'JAX/{nuts_sampler} not found, falling back to the pymc sampler')
                nuts_sampler = 'pymc'

            with hydro_model:
                if find_map:
                    self.logger.info('Running find_MAP to initialize the sampler...')
                    start = pm.find_MAP(method='L-BFGS-B', maxeval=5000)
                else:
                    start = None

                trace = pm.sample(num_samples, tune=num_warmup, chains=num_chains, target_accept=target_accept, nuts_sampler=nuts_sampler,
                                  initvals=start)

                # Log likelihood for LOO/WAIC model comparison. PyMC cannot compute it on a model with initval set (pm_priors uses
                # initval=start), and the initial values are no longer needed after sampling
                for rv in hydro_model.free_RVs:
                    hydro_model.set_initval(rv, None)
                pm.compute_log_likelihood(trace)

            az.to_netcdf(trace, trace_path)
            self.logger.info(f'InferenceData saved to {trace_path}')


            # trace = az.from_netcdf(trace_path)


            self.delta_results = calc_delta_quantities(self, trace, dict_utils, model_mstar=model_mstar, mstar=mstar, deltas=deltas,
                                                       outfile=os.path.join(self.fit_dir, f'{model_tag}_results.dat'))

            # Everything needed for the plots; several fitted Mhyd objects can be plotted together with hydromass.show_fit.plot_fits
            self.fit_products = compute_products(self, trace, dict_utils, model_tag, models_mass, model_mstar=model_mstar, mstar=mstar,
                                                 model_missing_mass=model_missing_mass, fit_SB=fit_SB, fit_kT=fit_kT, fit_Z_Ab=fit_Z_Ab,
                                                 fit_SZ=fit_SZ, fit_bkg=fit_bkg, delta_results=self.delta_results)

            save_products(self.fit_products, os.path.join(self.fit_dir, f'{model_tag}_products.fits'))

            if auto_plot:
                plot_likelihood([self.fit_products], self.fit_dir, suffix=f'_{model_tag}')
                plot_mass([self.fit_products], outfile=os.path.join(self.fit_dir, f'mass_{model_tag}.pdf'))
                plot_mass_fractions([self.fit_products], outfile=os.path.join(self.fit_dir, f'mass_fractions_{model_tag}.pdf'))
                plot_corner([self.fit_products], outfile=os.path.join(self.fit_dir, f'corner_{model_tag}.pdf'))
                plot_thermodynamics([self.fit_products], outfile=os.path.join(self.fit_dir, f'thermodynamics_{model_tag}.pdf'))

    def fit_forward(self,
            fit_SB=False, fit_bkg=False, bkglim_amin=None, backval=None, quick_density=True,
            fit_Z_Ab=False, model_Z_Ab=None,
            fit_kT=False,
            fit_SZ=False,
            fit_eta=False,
            model_forward=None,
            sb_setup=None,
            rmin_amin=None, rmax_amin=None,
            num_warmup=1000, num_samples=2000, target_accept=0.95, num_chains=4, nuts_sampler="numpyro",
            find_map=True, deltas=(2500, 1000, 500, 200), auto_plot=True, use_clip=True,
            force_positive_mass=False, force_convective_stability=False, force_increasing_mass=False,
                    nmore=5
            ):
        '''
        CLAUDE: Same as :meth:`fit_massmod`, but with a parametric forward model of the 3D pressure instead of mass models
        (see :func:`hydromass.likelihoods.kt_from_forward`). The total mass is computed from the hydrostatic equilibrium equation,
        so the mass plot shows the total and gas mass only. Outputs go to <dir>/<model_tag>, with model_tag e.g. FORWARD_SB_kT.

        :param quick_density:
            Whether to run :meth:`quick_fit_density` before the fit, to use its density coefficients (and BKG) as starting values. Default True.
        :param model_forward:
            Forward pressure model, e.g. :class:`hydromass.model_forward.ForwardModel`. Default None: ForwardModel() with its default priors.
        :param use_clip:
            As in :meth:`fit_massmod`. Default True.
        :param force_positive_mass:
            Soft penalty against negative mass, i.e. - d ln P / d ln r < 0. Default False.
        :param force_convective_stability:
            Soft penalty against an entropy decreasing with radius. Default False.
        :param force_increasing_mass:
            Soft penalty against a mass decreasing with radius. Default False.

        The other parameters are as in :meth:`fit_massmod`.
        '''
        # All the arguments of fit, stored in setup.log by store_fit_setup
        fit_args = {k: v for k, v in locals().items() if k != 'self'}

        if self.tag is None:
            # Name of the fit, e.g. FORWARD_SB_kT, used in plot names and legends
            model_tag = 'FORWARD'
            if fit_SB:
                model_tag += '_SB'
            if fit_kT:
                model_tag += '_kT'
            if fit_SZ:
                model_tag += '_SZ'
                if fit_eta:
                    model_tag += '_eta'
            if fit_Z_Ab:
                model_tag += '_Z'
            if self.conversion_factor_type != 'table':
                model_tag += '_fixedccf'
        else:
            model_tag = self.tag

        # All the outputs of this fit go to a folder named after the model
        self.fit_dir = os.path.join(self.dir, model_tag)
        os.makedirs(self.fit_dir, exist_ok=True)

        trace_path = os.path.join(self.fit_dir, f'trace_{model_tag}.nc')
        if not os.path.isfile(trace_path):
            self.store_fit_setup(fit_args)

            if (fit_SB is False) and (fit_kT is False) and (fit_Z_Ab is False) and (fit_SZ is False):
                raise ValueError('Nothing to do. At least one of [fit_SB, fit_kT, fit_Z_Ab, fit_SZ] must be set to True')

            if fit_Z_Ab and model_Z_Ab is None:
                raise ValueError('fit_Z_Ab is True but model_Z_Ab is not set')

            dict_utils = common_utils(self, rmin_amin=rmin_amin, rmax_amin=rmax_amin, nmore = nmore)

            sb_utils(self, dict_utils, sb_setup=sb_setup, fit_bkg=fit_bkg, bkglim_amin=bkglim_amin, backval=backval)

            if quick_density:
                self.quick_fit_density(dict_utils, fit_bkg=fit_bkg)

            if fit_kT:
                spec_utils(self, dict_utils)

            hydro_model = pm.Model()
            with hydro_model:
                params = dict()

                get_density(params, dict_utils, self, fit_bkg=fit_bkg)

                get_metal(params, model_Z_Ab, self, dict_utils, use_clip=use_clip)

                if fit_Z_Ab:
                    likelihood_metal(params, model_Z_Ab, self, dict_utils)

                kt_from_forward(params, model_forward, self, dict_utils, force_positive_mass=force_positive_mass,
                                force_convective_stability=force_convective_stability, force_increasing_mass=force_increasing_mass,
                                use_clip=use_clip)

                if fit_kT:
                    likelihood_kt(params, self, dict_utils)

                if fit_SZ:
                    likelihood_sz(params, self, dict_utils, fit_eta=fit_eta)

                get_ccf(params, self, dict_utils, self.conversion_factor_type)

                if fit_SB:
                    likelihood_sb(params, dict_utils, self, fit_bkg=fit_bkg)

            self.logger.info('Starting the sampler...')

            if nuts_sampler in ('numpyro', 'blackjax') and not isjax:
                self.logger.info(f'JAX/{nuts_sampler} not found, falling back to the pymc sampler')
                nuts_sampler = 'pymc'

            with hydro_model:
                if find_map:
                    self.logger.info('Running find_MAP to initialize the sampler...')
                    start = pm.find_MAP(method='L-BFGS-B', maxeval=5000)
                else:
                    start = None

                trace = pm.sample(num_samples, tune=num_warmup, chains=num_chains, target_accept=target_accept, nuts_sampler=nuts_sampler,
                                  initvals=start)

                # Log likelihood for LOO/WAIC model comparison. PyMC cannot compute it on a model with initval set (pm_priors uses
                # initval=start), and the initial values are no longer needed after sampling
                for rv in hydro_model.free_RVs:
                    hydro_model.set_initval(rv, None)
                pm.compute_log_likelihood(trace)

            az.to_netcdf(trace, trace_path)
            self.logger.info(f'InferenceData saved to {trace_path}')

            # trace = az.from_netcdf(trace_path)
            self.delta_results = calc_delta_quantities(self, trace, dict_utils, deltas=deltas,
                                                       outfile=os.path.join(self.fit_dir, f'{model_tag}_results.dat'))

            # No mass models: the mass products are the total (hydrostatic) and gas mass; the corner plot shows the forward model parameters
            self.fit_products = compute_products(self, trace, dict_utils, model_tag, [], fit_SB=fit_SB, fit_kT=fit_kT, fit_Z_Ab=fit_Z_Ab,
                                                 fit_SZ=fit_SZ, fit_bkg=fit_bkg, delta_results=self.delta_results)
            self.fit_products['corner'] = _corner_inputs(trace, [model_forward], dict_utils)

            save_products(self.fit_products, os.path.join(self.fit_dir, f'{model_tag}_products.fits'))

            if auto_plot:
                plot_likelihood([self.fit_products], self.fit_dir, suffix=f'_{model_tag}')
                plot_mass([self.fit_products], outfile=os.path.join(self.fit_dir, f'mass_{model_tag}.pdf'))
                plot_mass_fractions([self.fit_products], outfile=os.path.join(self.fit_dir, f'mass_fractions_{model_tag}.pdf'))
                plot_corner([self.fit_products], outfile=os.path.join(self.fit_dir, f'corner_{model_tag}.pdf'))
                plot_thermodynamics([self.fit_products], outfile=os.path.join(self.fit_dir, f'thermodynamics_{model_tag}.pdf'))
