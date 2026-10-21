import logging
import sys
import os

import ipdb
import numpy as np
import time
import pytensor.tensor as tt


isjax = False

try:
    import pymc.sampling.jax as pmjax

except ImportError:
    print('JAX not found, using default sampler')

else:
    print('We will use the JAX sampler')
    isjax = True
    import pymc.sampling.jax as pmjax

import pymc as pm
import arviz as az

from .plots import cumsum_mat, estimate_P0, kt_from_samples, P_from_samples, mgas_pm
from .nonparametric import Run_NonParametric_PyMC3
from .forward import Run_Forward_PyMC3
from .deproject import (calc_density_operator, list_params, calc_linear_operator, calc_sb_operator, list_params_density, MyDeprojVol,
                        calc_density_operator_pm, elongation_correction)
from .constants import cgsamu, cgskpc, Msun, y_prefactor, kev2erg, cgsMpc, cgsG
from .emissivity import calc_emissivity, variable_ccf
from .pnt import get_data_file_path, alpha_turb_pm, NPmodel
from .polytropic import Run_Polytropic_PyMC3
from .mean_molecular_weights import mean_molecular_weights
from .save import SaveModel, SaveForward, SaveGP
from .wl import WLmodel
# from .utility_old import sb_utils, rads_more
from .pyutils import create_logger, warning
from .conversion_factor_emissivity import calc_emissivity, ConversionFactor, FitCCFTable
from .utility import sb_utils, common_utils, spec_utils, mass_utils, ccf_utils, forw_utils, gp_utils
from .likelihoods import get_density, likelihood_sb, get_metal, likelihood_metal, kt_from_mass, likelihood_kt, get_ccf, kt_from_forw, kt_from_gp


__all__ = ['Run_Mhyd_PyMC3', 'Mhyd']

def Run_Mhyd_PyMC3(Mhyd, model,bkglim=None,nmcmc=1000,fit_bkg=False,back=None,
                   samplefile=None,
                   nrc=None, min_rc=None, max_rc=None,
                   nbetas=6,min_beta=0.6, max_beta=3,
                   nmore=5,
                   p0_prior: tuple  | None=None, tune=500, dmonly=False, mstar=None, find_map=True,
                   pnt=False, pnt_model='Ettori', rmin=0., rmax=None, p0_type='sb', init='ADVI', target_accept=0.9,
                   fit_elong=False, use_jax=True, wlonly=False, pnt_prior='sim', fit_eta=False, fix_beta_pnt=False, do_ppc=False,
                   cM_relation = False):
    """

    Set up hydrostatic mass model and optimize with PyMC3. The routine takes a parametric mass model as input and integrates the hydrostatic equilibrium equation to predict the 3D pressure profile:

    .. math::

        P_{3D}(r) = P_0 + \\int_{r}^{r_0} \\rho_{gas}(r) \\frac{G M_{mod}(<r)}{r^2} dr

    with :math:`r_0` the outer radial boundary of the input data and :math:`P_0` the pressure at :math:`r_0`.

    The gas density profile is fitted to the surface brightness profile and described as a linear combination of King functions. The mass model should be defined using the :class:`hydromass.functions.Model` class, which implements a number of popular mass models. The 3D mass profile is then projected along the line of sight an weighted by spectroscopic-like weights to predict the spectroscopic temperature profile.

    The parameters of the mass model and of the gas density profile are fitted jointly to the data. Priors on the input parameters can be set by the user in the definition of the mass model.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object including the loaded data and initial setup (mandatory input)
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param model:  A :class:`hydromass.functions.Model` object including the chosen mass model and its input values (mandatory input)
    :type model: class:`hydromass.functions.Model`
    :param bkglim: Limit (in arcmin) out to which the SB data will be fitted; if None then the whole range is considered. Defaults to None.
    :type bkglim: float
    :param nmcmc: Number of PyMC3 steps. Defaults to 1000
    :type nmcmc: int
    :param fit_bkg: Choose whether the counts and the background will be fitted on-the-fly using a Poisson model (fit_bkg=True) or if the surface brightness will be fitted, in which case it is assumed that the background has already been subtracted and Gaussian likelihood will be used (default = False)
    :type fit_bkg: bool
    :param back: Input value for the background. If None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True. Defaults to None.
    :type back: float
    :param samplefile: Name of ASCII file to output the final PyMC3 samples
    :type samplefile: str
    :param nrc: Number of core radii values to set up the multiscale model. Defaults to the number of data points / 4
    :type nrc: int
    :param nbetas: Number of beta values to set up the multiscale model (default = 6)
    :type nbetas: int
    :param min_beta: Minimum beta value (default = 0.6)
    :type min_beta: float
    :param nmore: Number of points to the define the fine grid onto which the mass model and the integration are performed, i.e. for one spectroscopic/SZ value, how many grid points will be defined. Defaults to 5.
    :type nmore: int
    :param p0_prior: Set of two values defining the mean and standard deviation of the Gaussian prior on p0. If None, the code attempts to determine the value of P0 using the :func:`hydromass.plots.estimate_P0` function, which fits a rough gNFW function to estimate the shape of the pressure profile and uses the fitted function to approximate the value of P0.
    :type p0_prior: numpy.ndarray
    :param tune: Number of NUTS tuning steps. Defaults to 500
    :type tune: int
    :param dmonly: Specify whether the mass model is fitted to the total mass (dmonly=False) or to the dark matter only after subtracting the gas mass and the stellar mass if provided (dmonly=True). Defaults to False.
    :type dmonly: bool
    :param mstar: If dmonly=True, provide an array containing the cumulative stellar mass profile, which will be subtracted when adjusting the mass model to the dark matter only.
    :type mstar: numpy.ndarray
    :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
    :type find_map: bool
    :param pnt: Attempt to model the non-thermal pressure profile. If pnt=True, the non-thermal pressure profile of Angelinelli et al. 2020 and the corresponding parameters are used to marginalize over the impact of non-thermal pressure. Defaults to False.
    :type pnt: bool
    :param rmin: Minimum limiting radius (in arcmin) of the active region for the surface brightness. If rmin=None, no minimum radius is applied. Defaults to None.
    :type rmin: float
    :param rmax: Maximum limiting radius (in arcmin) of the active region for the surface brightness. If rmax=None, no maximum radius is applied. Defaults to None.
    :type rmax: float
    :param p0_type: For the estimation of P0, choose whether we will use the surface brightness profile (p0_type='sb') or the spectral normalization (p0_type='norm'). Defaults to 'sb'.
    :type p0_type: str
    :param init: Choose the initialization method for PyMC. Defaults to 'ADVI'
    :type init: str
    :param target_accept: Set the target_accept parameter for PyMC. Defaults to 0.9
    :type target_accept: float
    :param pnt_model: Choose the analytic model for non-thermal pressure modeling. Available choices are 'Angelinelli' (Angelinelli+20) and 'Ettori' (Ettori\&Eckert22). Defaults to 'ettori'
    :type pnt_model: str
    :param fit_elong: Set whether the elongation of the system along the line of sight as a free model parameter to account for differences between various observables induced by line-of-sight elongation. Defaults to False
    :type fit_elong: bool
    :param use_jax: Use JAX optimization when sampling using the numpyro NUTS implementation. Defaults to True
    :type use_jax: bool
    :param wlonly: Set whether the fit will be done to weak lensing data only. Defaults to False
    :type wlonly: bool
    :param pnt_prior: Choose whether informative priors from simulations will be applied to the Pnt profile (pnt_prior='sim'). Alternatively, flat priors will be adopted. Defaults to 'sim'.
    :type pnt_prior: str
    :param fit_eta: Fit for $\\eta_T = P_X/P_{SZ}$ according to the definition of Kozmanyan et al. 2019. Defaults to False.
    :type fit_eta: bool
    :param fix_beta_pnt: When fitting for non-thermal pressure, fix the $\\beta_{NT}$ parameter (i.e. the slope of the polytropic relation) to the typical value of 0.9 (Ettori & Eckert 2022). Defaults to False.
    :type fix_beta_pnt: bool
    """


    if fit_elong and fit_eta:

        Mhyd.logger.info('Elongation and eta are completely degenerate, it is not possible to fit for both at the same time. Aborting.')
        return

    # prof = Mhyd.sbprof
    # sb = prof.profile.astype('float32')
    # esb = prof.eprof.astype('float32')
    # rad = prof.bins.astype('float32')
    # erad = prof.ebins.astype('float32')
    # if prof.counts is not None:
    #     counts = prof.counts.astype('int32')
    #     bkgcounts = prof.bkgcounts.astype('float32')
    # else:
    #     if fit_bkg:
    #         print('The fit_bkg option can only be used when fitting counts, which are not available. Reverting to default')
    #         fit_bkg = False
    #
    # if not prof.voronoi:
    #     area = prof.area.astype('float32')
    #     exposure = prof.effexp.astype('float32')
    # nbin = prof.nbin
    #
    # nmin = 0
    # nmax = len(sb)
    #
    # if rmax is None:
    #     rmax = np.max(rad+erad)
    #
    # if rmin is None:
    #     rmin = 0
    #
    # valid = np.where(np.logical_and(rad>=rmin, rad<rmax))


#    if rmin is not None:
#        valid = np.where(rad>=rmin)
#        sb = sb[valid]
#        esb = esb[valid]
#        rad = rad[valid]
#        erad = erad[valid]
#        counts = counts[valid]
#        area = area[valid]
#        exposure = exposure[valid]
#        bkgcounts = bkgcounts[valid]
#        nmin = valid[0][0]
#
#    if rmax is not None:
#        valid = np.where(rad <= rmax)
#        sb = sb[valid]
#        esb = esb[valid]
#        rad = rad[valid]
#        erad = erad[valid]
#        counts = counts[valid]
#        area = area[valid]
#        exposure = exposure[valid]
#        bkgcounts = bkgcounts[valid]
#        valori = np.where(prof.bins <= rmax)
#        nmax = np.max(valori[0])+1

    # nbin = len(sb)
    #
    #
    # # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    # if bkglim is None:
    #     bkglim=float(np.max(rad+erad))
    #     Mhyd.bkglim = bkglim
    #     if back is None:
    #         back = sb[len(sb) - 1]
    # else:
    #     Mhyd.bkglim = bkglim
    #     backreg = np.where(rad>bkglim)
    #     if back is None:
    #         back = np.mean(sb[backreg])
    #
    # # Set source region
    # sourcereg = np.where(rad < bkglim)
    #
    # # Set vector with list of parameters
    # pars = list_params(rad, sourcereg, nrc, nbetas, min_beta)
    #
    # Mhyd.pars = pars
    #
    # Mhyd.sourcereg = sourcereg
    #
    # npt = len(pars)
    #
    # if prof.psfmat is not None:
    #     psfmat = prof.psfmat[nmin:nmax,nmin:nmax]
    # else:
    #     psfmat = np.eye(len(sb))
    #
    # # Compute linear combination kernel
    # if fit_bkg:
    #
    #     K = calc_linear_operator(rad, sourcereg, pars, area, exposure, np.transpose(psfmat)) # transformation to counts
    #
    # else:
    #     Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)
    #
    #     K = np.dot(psfmat, Ksb)
    #     # K = calc_sb_operator_psf(rad, sourcereg, pars, area, exposure, psfmat) # transformation to surface brightness
    #
    # # Set up initial values
    # if np.isnan(sb[0]) or sb[0] <= 0:
    #     testval = -10.
    # else:
    #     testval = np.log(sb[0] / npt)
    # if np.isnan(back) or back <= 0 or back is None:
    #     testbkg = -10.
    # else:
    #     testbkg = np.log(back)
    #
    # z = Mhyd.redshift
    #
    # transf = Mhyd.transf
    #
    # pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, nrc, nbetas, min_beta)
    #
    # if fit_bkg:
    #
    #     Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc)
    #
    # else:
    #
    #     Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc, withbkg=False)
    #
    # # Define the fine grid onto which the mass model will be computed
    # rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)
    #
    # rref_m = (rin_m + rout_m)/2.
    # vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)
    #
    # vol = vx.deproj_vol().T
    #
    # Mhyd.cf_prof = None
    #
    # try:
    #     nn = len(Mhyd.ccf)
    #
    # except TypeError:
    #
    #     print('Single conversion factor provided, we will assume it is constant throughout the radial range')
    #
    #     cf = Mhyd.ccf
    #
    # else:
    #
    #     tcf = Mhyd.ccf[nmin:nmax]
    #
    #     if len(tcf) != len(rad):
    #
    #         print('The provided conversion factor has a different length as the input radial binning. Adopting the mean value.')
    #
    #         cf = np.mean(Mhyd.ccf)
    #
    #     else:
    #
    #         print('Interpolating conversion factor profile onto the radial grid')
    #
    #         cf = np.interp(rref_m, rad * Mhyd.amin2kpc, tcf)
    #
    #         Mhyd.cf_prof = cf
    #
    # if Mhyd.spec_data is not None:
    #
    #     if Mhyd.spec_data.psfmat is not None:
    #
    #         mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)
    #
    #         proj_mat = np.dot(mat1, vol)
    #
    #     else:
    #
    #         proj_mat = np.dot(sum_mat, vol)
    #
    #
    # if fit_bkg:
    #
    #     Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)
    #
    # else:
    #
    #     Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

    (testval, testbkg, npt, bkgcounts, counts, sb, esb, valid, cf, rin_m, rout_m, rref_m, proj_mat, volmat, index_sz, ntm, rad, rmin, rmax,
     nbin) = sb_utils(Mhyd, fit_bkg = fit_bkg, rmin = rmin, rmax = rmax, bkglim = bkglim, back = back,
                      nbetas = nbetas, min_beta = min_beta, max_beta = max_beta, nrc = nrc, min_rc = min_rc, max_rc = max_rc,
                      nmore = nmore)
    # np.savez('rout', rout_m=rout_m, rin_m=rin_m, ntm=ntm, index_sz=index_sz, psfmat = Mhyd.sz_data.psfmat)
    # sys.exit()

    dlogr = (np.log(rout_m) - np.log(rin_m))

    if dmonly and mstar is not None:

        r_mstar = mstar[:, 0]

        cum_mstar = mstar[:, 1]

        mstar_m = np.interp(rref_m, r_mstar, cum_mstar)

    nptmore = len(rout_m)

    int_mat = cumsum_mat(nptmore)

    hydro_model = pm.Model()

    if pnt:

        if model.massmod not in ['NFW'] and pnt_model in ['Angelinelli']:

            Mhyd.logger.info('Angelinelli non-thermal pressure correction is currently implemented only for NFW, reverting to thermal only')

            pnt = False

        else:

            file_means = get_data_file_path('pnt_mean_log.dat')

            file_cov = get_data_file_path('pnt_covmat_log.dat')

            pnt_mean = np.loadtxt(file_means).astype(np.float32)

            pnt_cov = np.loadtxt(file_cov).astype(np.float32)

    if Mhyd.veldata is not None and not pnt:

        Mhyd.logger.info('Warning: velocity data were provided but pnt is set to False. The velocity data will be ignored.')

    if cM_relation:

        import jax.numpy as jnp
        from colossus.halo import concentration
        from colossus.cosmology import cosmology


        # sigma8 = 0.811
        # ns = 0.965
        # colossus_cosmo = cosmology.fromAstropy(
        #     Mhyd.cosmo,
        #     sigma8=sigma8,
        #     ns=ns,
        #     cosmo_name='custom_from_astropy'
        # )
        #
        #
        # colossus_cosmo.matterPowerSpectrumModel = 'eisenstein98'

        cosmology.setCosmology('planck18')

        from pytensor.graph.op import Op
        from pytensor.graph.basic import Apply
        from pytensor.link.jax.dispatch import jax_funcify
        import jax
        import jax.numpy as jnp
        from scipy.interpolate import interp1d
        from colossus.halo import concentration

        class ColossusIshiyamaGridOp(Op):
            def __init__(self, m_min=1e13, m_max=5e15, z_val=0.0, num_m=500, cosmo_h=0.6774):
                super().__init__()
                # 1. Pre-compute log-space grid using Colossus
                self.m_grid = np.logspace(np.log10(m_min), np.log10(m_max), num_m)
                self.c_grid = concentration.concentration(
                    self.m_grid * cosmo_h,
                    mdef='500c',
                    z=z_val,
                    model='ishiyama21'
                )

                self.log_m_grid = np.log10(self.m_grid)
                self.log_c_grid = np.log10(self.c_grid)

                # CPU 1D Interpolator (for non-JAX PyMC sampling)
                self.cpu_interp = interp1d(
                    self.log_m_grid,
                    self.log_c_grid,
                    kind='linear',
                    fill_value='extrapolate'
                )

            def make_node(self, m500):
                m500 = tt.as_tensor_variable(m500)
                return Apply(self, [m500], [m500.type()])

            def perform(self, node, inputs, outputs):
                m500_val = inputs[0]
                log_m = np.log10(m500_val)
                log_c = self.cpu_interp(log_m)
                outputs[0][0] = np.array(10 ** log_c, dtype=np.float64)

            def grad(self, inputs, g_outputs):
                (m500,) = inputs
                (g_out,) = g_outputs

                # Finite difference gradient on the interpolated grid for PyMC C-backend
                grad_op = ColossusIshiyamaGridGradOp(self)
                return [g_out * grad_op(m500)]

        class ColossusIshiyamaGridGradOp(Op):
            def __init__(self, main_op):
                super().__init__()
                self.main_op = main_op

            def make_node(self, m500):
                m500 = tt.as_tensor_variable(m500)
                return Apply(self, [m500], [m500.type()])

            def perform(self, node, inputs, outputs):
                m500_val = inputs[0]
                eps = m500_val * 1e-5
                c_plus = 10 ** self.main_op.cpu_interp(np.log10(m500_val + eps))
                c_minus = 10 ** self.main_op.cpu_interp(np.log10(m500_val - eps))
                outputs[0][0] = np.array((c_plus - c_minus) / (2 * eps), dtype=np.float64)

        # 2. Register pure JAX dispatch (Runs entirely on JAX/XLA at maximum speed)
        @jax_funcify.register(ColossusIshiyamaGridOp)
        def jax_funcify_ColossusIshiyamaGridOp(op, **kwargs):
            log_m_grid_jax = jnp.array(op.log_m_grid)
            log_c_grid_jax = jnp.array(op.log_c_grid)

            def jax_colossus_interp(m500):
                log_m = jnp.log10(m500)
                log_c = jnp.interp(log_m, log_m_grid_jax, log_c_grid_jax)
                return 10 ** log_c

            return jax_colossus_interp

        colossus_grid_op = ColossusIshiyamaGridOp(
            m_min=1e13,
            m_max=5e15,
            z_val=Mhyd.redshift,
            num_m=500,
            cosmo_h=Mhyd.cosmo.h
        )



    with hydro_model:
        # Model parameters


        initvals = dict()

        pymc_vars = dict()

        allpmod = []

        print(model.massmod)

        for i in range(model.npar):

            name = model.parnames[i]

            if not model.fix[i]:

                lim = model.limits[i]

                if model.massmod == 'EIN3':

                    Mhyd.logger.info(f'pm.TruncatedNormal({name}, mu={model.start[i]}, sigma={model.sd[i]}, lower={lim[0]}, upper={lim[1]})')

                    pymc_vars[name] = pm.TruncatedNormal(name, mu=model.start[i], sigma=model.sd[i], lower=lim[0], upper=lim[1])

                else:

                    Mhyd.logger.info(f'pm.Uniform({name}, lower={lim[0]}, upper={lim[1]})')

                    pymc_vars[name] = pm.Uniform(name, lower=lim[0], upper=lim[1])

                    initvals[name] = model.start[i]

            else:

                Mhyd.logger.info(f'pm.Deterministic({name}, pm.math.constant({model.start[i]}))')

                pymc_vars[name] = pm.Deterministic(name, pm.math.constant(model.start[i]))

                # modpar = pm.ConstantDist(name, model.start[i]) # This does not work anymore in pymc

            allpmod.append(pymc_vars[name])

        pmod = pm.math.stack(allpmod, axis=0)

        if fit_elong:
            # Prior on the line-of-sight elongation parameter
            elongation = pm.TruncatedNormal('elong', mu=1.0, sigma=0.2, lower=0.1, upper=10)
            Mhyd.logger.info("pm.TruncatedNormal('elong', mu=1, sigma=2, lower=0.1, upper=10)")
        else:
            elongation = 1

        if fit_eta:
            Mhyd.logger.info("pm.Uniform('eta', lower=0.2, upper=5)")
            eta = pm.Uniform('eta', lower=0.2, upper=5)
        else:
            eta = 1


        if cM_relation:
            if model.massmod == 'NFW':
                Mhyd.logger.info('Using concentration mass relation (Ishiyama21)')

                c500 = pymc_vars['cdelta']
                r500 = pymc_vars['rdelta']
                m500 = 500 * Mhyd.rho_cz * 4 / 3 * np.pi * (r500/1000)**3
                c500c_ishiyama = colossus_grid_op(m500)


                pm.Potential(
                    "cM_relation",
                    pm.logp(
                        pm.Normal.dist(mu = pm.math.log(c500c_ishiyama), sigma = 0.23),
                        pm.math.log(c500)
                    )
                )

        if not wlonly:
            # Priors for unknown model parameters
            if True:
                coefs = pm.Normal('coefs', mu=testval, sigma=20, shape=npt)
            else:

                sigma_w = 20#pm.HalfNormal("sigma_w", 10.0)
                corr_len = 1 #pm.HalfNormal("corr_len", 1)

                # ell_rc = pm.HalfNormal("ell_rc", 1.0)
                # ell_beta = pm.HalfNormal("ell_beta", 1.0)
                # #
                # # dist_pars = Mhyd.dist_rc**2 / ell_rc**2 + Mhyd.dist_beta**2 / ell_beta**2
                #
                # dist2 = (
                #         Mhyd.dist_rc2 / ell_rc ** 2 +
                #         Mhyd.dist_beta2 / ell_beta ** 2
                # )

                # cov_w = sigma_w ** 2 * tt.exp(-0.5 * Mhyd.dist_pars2/corr_len**2)

                # cov_w += 1e-6 * tt.eye(npt)

                coefs = pm.MvNormal("coefs", mu = tt.constant(testval), cov = Mhyd.cov_w)

            if fit_bkg:

                bkgd = pm.Normal('bkg', mu=testbkg, sigma=4, shape=1) # in case fit_bkg = False this is not fitted

                ctot = pm.math.concatenate((coefs, bkgd), axis=0)

                al = pm.math.exp(ctot)

                pred = pm.math.dot(Mhyd.K, al) + bkgcounts  # Predicted number of counts per annulus

            else:

                al = pm.math.exp(coefs)

                pred = pm.math.dot(Mhyd.K, al)

            # Integration constant as a fitting (nuisance) parameter
            if p0_prior is not None:

                P0_est = p0_prior[0]

                err_P0_est = p0_prior[1]

            else:

                P0_est = estimate_P0(Mhyd=Mhyd, dens=p0_type)

                Mhyd.logger.info(f'Estimated value of P0: {P0_est}')

                err_P0_est = np.log(10) * P0_est #P0_est # 1 in ln

            logp0 = pm.TruncatedNormal('logp0', mu=np.log(P0_est), sigma=err_P0_est / P0_est,
                                       lower=np.log(P0_est) - 2*np.log(10),
                                       upper=np.log(P0_est) + 2*np.log(10))

            Mhyd.logger.info(f"pm.TruncatedNormal('logp0', mu={np.log(P0_est)}, sigma={err_P0_est / P0_est}, " #+
                  f"lower={np.log(P0_est) - 2*np.log(10)}, upper={np.log(P0_est) + 2*np.log(10)})")

            if pnt :

                if pnt_model=='Angelinelli':

                    if pnt_prior=='sim':
                        pnt_pars = pm.MvNormal('Pnt', mu=pnt_mean, cov=pnt_cov, shape=(1,3))

                    else:
                        a0 = pm.Uniform('a0', lower=-0.5, upper=2.0)
                        a1 = pm.Uniform('a1', lower=-1.0, upper=2.0)
                        a2 = pm.Uniform('a2', lower=-6., upper=0.)

                        pnt_pars = tt.stack([a0, a1, a2], axis=0).reshape((1, 3))

                    #a2 = pnt_pars[0, 2]

                    #pm.Potential("bound", tt.switch(a2 > 0, 0, -np.inf))

                    #pnt_pars = pm.MvNormal('Pnt', mu=pnt_mean[:2], cov=pnt_cov[:2, :2], shape=(2,))
                    #a0 = pnt_pars[0, 0]
                    #a1 = pnt_pars[0, 1]
                    #a2 = pm.math.exp(pnt_pars[0, 2])

                    # Log-normal prior for a2
                    #mu_a2 = np.log(7.25e-2)
                    #sigma_a2 = np.sqrt(3.44e-3)
                    #a2 = pm.Lognormal('a2', mu=mu_a2, sigma=sigma_a2)

                    #pnt_pars = tt.stack([a0, a1, a2], axis=0).reshape((1, 3))

                if pnt_model=='Ettori':

                    if not fix_beta_pnt:
                        beta_nt = pm.Normal('beta_nt', mu=0.9, sigma=0.13) #0.13

                    else:
                        beta_nt = pm.Deterministic('beta_nt', pm.math.constant(0.9))

                    logp0_nt = pm.Uniform('p0_nt', lower=-5, upper=-2.) #-2

                    pnt_pars = [beta_nt, logp0_nt]

            #for RV in hydro_model.basic_RVs:
            #    print(RV.name, RV.logp(hydro_model.test_point))

            press00 = pm.math.exp(logp0)

            if fit_elong and not fit_bkg:

                Kdens_t = calc_density_operator_pm(rref_m / Mhyd.amin2kpc, Mhyd.pardens, elongation, Mhyd.amin2kpc)

                Kdens_t_out = calc_density_operator_pm(rout_m / Mhyd.amin2kpc, Mhyd.pardens, elongation, Mhyd.amin2kpc)

                dens_m = pm.math.sqrt(pm.math.dot(Kdens_t, al) / cf * Mhyd.transf)  # electron density in cm-3

                dens_m_out = pm.math.sqrt(pm.math.dot(Kdens_t_out, al) / cf * Mhyd.transf)  # electron density in cm-3

            else:

                dens_m = pm.math.sqrt(pm.math.dot(Mhyd.Kdens_m, al) / cf * Mhyd.transf)  # electron density in cm-3

                dens_m_out = pm.math.sqrt(pm.math.dot(Mhyd.Kdens_m_out, al) / cf * Mhyd.transf)  # electron density in cm-3

            mbar_rref = 0
            if dmonly:

                nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

                mgas_out = mgas_pm(rin_m, rout_m, dens_m) * nhconv / Mhyd.mfact0

                # Assuming mgas(rref) = (mgas(rin)+mgas(rout)) / 2
                mgas_rref = tt.concatenate([[mgas_out[0]], (mgas_out[1:] + mgas_out[:-1])]) / 2

                # Add stellar mass if provided
                if mstar is not None:

                    mbar_rref = mgas_rref + mstar_m / Mhyd.mfact0 / 1e13

                else:

                    mbar_rref = mgas_rref


            # Evaluate mass model
            if model.massmod in ['MOND', 'GMOND', 'EMOND']:

                # MOND also needs baryon mass

                assert mbar_rref != 0, 'Cannot use MOND/GMOND/EMOND if baryons are not passed. Select dmonly=True and optionally pass mstar'

                mass = model.func_pm(rref_m, *pmod, mbar = mbar_rref * 1e13 * Mhyd.mfact0) / Mhyd.mfact0 / 1e13

            else:


                mass = Mhyd.mfact * model.func_pm(rref_m, *pmod, delta = model.delta) / Mhyd.mfact0




            tot_mass = mass + mbar_rref # This is wrong. Adding something at rref + something at rout

            # Pressure gradient
            dpres = - tot_mass / rref_m ** 2 * dens_m * (rout_m - rin_m)

            press_out = press00 - pm.math.dot(int_mat, dpres)  # directly returns press_out

            # Non-thermal pressure correction, if any
            if pnt:

                if pnt_model!= 'Angelinelli' and pnt_model!='Ettori':

                    Mhyd.logger.info(f'Invalid Pnt model {pnt_model}, neglecting Pnt correction')

                    pth = press_out

                    pnt_prof = 0

                else:
                    if pnt_model == 'Angelinelli':

                        c200 = pmod[0]

                        r200c = pmod[1]

                        alpha_turb = alpha_turb_pm(rref_m, r200c, c200, Mhyd.redshift, pnt_pars)

                        pth_test = press_out * (1. - alpha_turb)

                        pnt_test = press_out * alpha_turb

                    if pnt_model == 'Ettori' :

                        log_pnt = beta_nt * pm.math.log(dens_m * 1e3) + logp0_nt * np.log(10)

                        pth_test = press_out - pm.math.exp(log_pnt)

                        pnt_test = pm.math.exp(log_pnt)

                    pth = pm.math.switch(pth_test <= 0, 1e-10, pth_test)

                    pnt_prof = pm.math.switch(pnt_test <= 0, 1e-10, pnt_test)

            else:

                pth = press_out

                pnt_prof = 0

        if not wlonly:
            # Density Likelihood
            if fit_bkg:

                count_obs = pm.Poisson('counts', mu=pred, observed=counts) #counts likelihood

                sbmod = pred * elongation / Mhyd.sbprof.area / Mhyd.sbprof.effexp

                sb_obs = pm.Deterministic('sb', sbmod)  # Sx: only saved here

            else:

                sbmod = pred * elongation

                sbmod = pm.Deterministic('sb', sbmod)  # Sx: only saved here

                sb_obs = pm.Normal('sb_obs', mu=sbmod[valid], observed=sb[valid], sigma=esb[valid]) #Sx likelihood

            # Temperature model and likelihood
            if Mhyd.spec_data is not None:

                # Model temperature
                t3d = pth / dens_m_out

                # Mazzotta weights
                ei = dens_m_out ** 2 * t3d ** (-0.75)

                # Temperature projection
                flux = pm.math.dot(proj_mat, ei)

                tproj = pm.math.dot(proj_mat, t3d * ei) / flux

                rmin_spec = 0.
                rmax_spec = np.max(Mhyd.spec_data.rout_x_am)

                if rmin is not None:
                    rmin_spec = rmin

                if rmax is not None:
                    rmax_spec = rmax

                valspec = np.where(np.logical_and(Mhyd.spec_data.rref_x_am>=rmin_spec, Mhyd.spec_data.rref_x_am<=rmax_spec))

                pm.Deterministic('tproj', tproj)

                T_obs = pm.Normal('kt', mu=tproj[valspec], observed=Mhyd.spec_data.temp_x[valspec], sigma=Mhyd.spec_data.errt_x[valspec])  # temperature likelihood

            # SZ pressure model and likelihood
            if Mhyd.sz_data is not None:

                if Mhyd.sz_data.sz_fit in ['P', 'logP']: # Fitting the pressure

                    print(f'Pressure fit {Mhyd.sz_data.sz_fit}')

                    pfit = pth[index_sz]

                    if fit_elong:
                        pfit = pfit * elongation

                    if fit_eta:
                        pfit = pfit * eta # P_SZ = P_X * eta

                    pm.Deterministic('pfit', pfit)

                    if Mhyd.sz_data.sz_fit == 'P':
                        P_obs = pm.MvNormal('P', mu=pfit, observed=Mhyd.sz_data.pres_sz, cov=Mhyd.sz_data.covmat_sz)  # SZ pressure likelihood
                    else:
                        P_obs = pm.MvNormal('P', mu=pm.math.log(pfit), observed=Mhyd.sz_data.logp_sz, cov=Mhyd.sz_data.covmat_sz)  # SZ pressure likelihood

                if Mhyd.sz_data.sz_fit in ['Y', 'logY']:

                    print(f'Compton y fit {Mhyd.sz_data.sz_fit}')

                    nout = 2 * nmore

                    rout_m_p = np.append(rout_m, np.logspace(np.log10(np.max(rout_m) * 1.1), np.log10(10000.), nout))
                    rin_m_p = np.append(rin_m, rout_m_p[ntm - 1:ntm - 1 + nout])

                    rref_m_p = (rin_m_p + rout_m_p) / 2.

                    slope = (pm.math.log(pth[ntm - 1]) - pm.math.log(pth[ntm - nout])) / (
                            pm.math.log(rref_m[ntm - 1]) - pm.math.log(rref_m[ntm - nout]))

                    pth_out = pth[ntm - 1] * (rref_m_p[ntm:] / rref_m[ntm - 1]) ** slope

                    pth_p = pm.math.concatenate([pth, pth_out], axis=0)

                    deproj = MyDeprojVol(rin_m_p, rout_m_p) # r from kpc to cm

                    proj_vol = deproj.deproj_vol().T

                    area_proj = np.pi * (-(rin_m_p) ** 2 + (rout_m_p) ** 2)

                    integ = pm.math.dot(proj_vol, pth_p) / area_proj * cgskpc

                    y_num = y_prefactor * integ  # prefactor in cm2/keV

                    if fit_elong:
                        yfit = elongation_correction(y_num, (rin_m_p + rout_m_p)/2*cgskpc, index_sz, elongation).flatten()

                    elif fit_eta:
                        yfit = y_num[index_sz] * eta

                    else:
                        yfit = y_num[index_sz]

                    if Mhyd.sz_data.psfmat is not None:

                        yfit = pm.math.dot(Mhyd.sz_data.psfmat, yfit)

                    pm.Deterministic('yfit', yfit)

                    print(y_num.shape.eval(), yfit.shape.eval(), index_sz, Mhyd.sz_data.logy_sz.shape, Mhyd.sz_data.covmat_sz.shape)

                    if Mhyd.sz_data.sz_fit == 'Y':
                        Y_obs = pm.MvNormal('Y', mu=yfit, observed=Mhyd.sz_data.y_sz, cov=Mhyd.sz_data.covmat_sz)
                    else:
                        Y_obs = pm.MvNormal('Y', mu=pm.math.log(yfit), observed=Mhyd.sz_data.logy_sz, cov=Mhyd.sz_data.covmat_sz)

        if Mhyd.wl_data is not None:

            WLdata = Mhyd.wl_data

            gmodel, rm, ev = WLmodel(WLdata, model, pmod)

            gmodel_elong = elongation_correction(gmodel, rm, ev, elongation).flatten()

            # gmodel_elong, rm, ev = WLmodel_elong(WLdata, model, pmod, elongation)

            g_obs = pm.MvNormal('WL', mu=gmodel_elong, observed=WLdata.gplus, cov=WLdata.covmat)

        if Mhyd.veldata is not None and pnt:

            sigmav, ev = NPmodel(Mhyd=Mhyd,
                                 rref_m=rref_m,
                                 dens_m=dens_m,
                                 pnt=pnt_prof,
                                 proj_vol=volmat)

            sv_obs = pm.Normal('sigmav', mu=sigmav[ev], sigma=Mhyd.veldata.vtot_error, observed=Mhyd.veldata.vtot)


    tinit = time.time()

    Mhyd.logger.info('Running HMC...')

    with hydro_model:

        if find_map:

            start = pm.find_MAP(start=initvals)

            if not isjax or not use_jax:

                trace = pm.sample(nmcmc, init=init, initvals=start, tune=tune, target_accept=target_accept)
            else:

                trace = pmjax.sample_numpyro_nuts(nmcmc, initvals=start, tune=tune, target_accept=target_accept)

        else:

            if not isjax or not use_jax:

                trace = pm.sample(nmcmc, init=init, tune=tune, target_accept=target_accept)

            else:

                trace = pmjax.sample_numpyro_nuts(nmcmc, tune=tune, target_accept=target_accept)

        if do_ppc:
            if not wlonly:
                Mhyd.ppc_sb = pm.sample_posterior_predictive(trace, var_names=['sb'])

            if Mhyd.spec_data is not None and not wlonly:

                Mhyd.ppc_kt = pm.sample_posterior_predictive(trace, var_names=['kt'])

            if Mhyd.sz_data is not None and not wlonly:

                if Mhyd.sz_data.pres_sz is not None: # Fitting the pressure

                    Mhyd.ppc_sz = pm.sample_posterior_predictive(trace, var_names=['P'])

                elif Mhyd.sz_data.y_sz is not None: # Fitting the Compton y parameter

                    Mhyd.ppc_sz = pm.sample_posterior_predictive(trace, var_names=['Y'])

            if Mhyd.wl_data is not None:

                Mhyd.ppc_wl = pm.sample_posterior_predictive(trace, var_names=['WL'])

            if Mhyd.veldata is not None:

                Mhyd.ppc_vel = pm.sample_posterior_predictive(trace, var_names=['sigmav'])


    Mhyd.logger.info('Done.')

    tend = time.time()

    Mhyd.logger.info(f' Total computing time is: {(tend - tinit) / 60.} minutes')

    Mhyd.logger.info('Computing log_likelihood')

    with hydro_model:

        pm.compute_log_likelihood(trace)

    Mhyd.trace = trace

    Mhyd.hydro_model = hydro_model

    if not wlonly:
        # Get chains and save them to file
        chain_coefs = np.array(trace.posterior['coefs'])

        sc_coefs = chain_coefs.shape

        sampc = chain_coefs.reshape(sc_coefs[0]*sc_coefs[1], sc_coefs[2])

        if fit_bkg:

            sampb = np.array(trace.posterior['bkg']).flatten()

            samples = np.vstack((sampc.T, sampb)).T #np.append(sampc, sampb, axis=1)

        else:
            samples = sampc

        Mhyd.samples = samples
        nsamp = len(samples)

        if samplefile is not None:
            np.savetxt(samplefile, samples)
            np.savetxt(samplefile+'.par',np.array([Mhyd.pars.shape[0]/nbetas,nbetas,min_beta,nmcmc]),header='pymc3')

    # Compute output deconvolved brightness profile
    if fit_elong:
        elong = (np.array(trace.posterior['elong'])).flatten()
    else:
        elong = 1

    if not wlonly:
        if fit_bkg:
            # Ksb = calc_sb_operator(rad, Mhyd.sourcereg, Mhyd.pars)

            allsb = np.dot(Mhyd.Ksb, np.exp(samples.T))

            bfit = np.median(np.exp(samples[:, npt]))

            Mhyd.bkg = bfit

            allsb_conv = np.dot(Mhyd.sbprof.psfmat, allsb[:npt, :] + np.exp(samples[:, npt])[np.newaxis, :])

        else:
            # Ksb = calc_sb_operator(rad, Mhyd.sourcereg, Mhyd.pars, withbkg=False)

            allsb = np.dot(Mhyd.Ksb, np.exp(samples.T))

            allsb_conv = np.dot(Mhyd.K, np.exp(samples.T))

            if fit_elong:

                elong_mat = np.tile(elong, nbin).reshape(nbin,nsamp)

                allsb = allsb * elong_mat #** 0.5

                allsb_conv = allsb_conv * elong_mat #** 0.5


        pmc = np.median(allsb, axis=1)
        pmcl = np.percentile(allsb, 50. - 68.3 / 2., axis=1)
        pmch = np.percentile(allsb, 50. + 68.3 / 2., axis=1)
        Mhyd.sb_dec = pmc
        Mhyd.sb_dec_lo = pmcl
        Mhyd.sb_dec_hi = pmch

        pmc = np.median(allsb_conv, axis=1)
        pmcl = np.percentile(allsb_conv, 50. - 68.3 / 2., axis=1)
        pmch = np.percentile(allsb_conv, 50. + 68.3 / 2., axis=1)

        pmc = np.median(trace.posterior.sb, axis = (0, 1))
        pmcl = np.percentile(trace.posterior.sb, 50. - 68.3 / 2., axis = (0, 1))
        pmch = np.percentile(trace.posterior.sb, 50. + 68.3 / 2., axis = (0, 1))

        Mhyd.sb = pmc
        Mhyd.sb_lo = pmcl
        Mhyd.sb_hi = pmch

    Mhyd.dmonly = dmonly
    Mhyd.wlonly = wlonly
    Mhyd.mstar = mstar
    Mhyd.pnt = pnt

    if pnt and not wlonly:

        if pnt_model == 'Angelinelli':

            if pnt_prior=='sim':

                Mhyd.pnt_pars = np.array(trace.posterior['Pnt']).reshape(sc_coefs[0] * sc_coefs[1], 3)

            else:

                a0 = np.array(trace.posterior['a0']).flatten()
                a1 = np.array(trace.posterior['a1']).flatten()
                a2 = np.array(trace.posterior['a2']).flatten()

                Mhyd.pnt_pars = np.vstack([a0,a1,a2]).T

            Mhyd.pnt_model = 'Angelinelli'

        if pnt_model == 'Ettori':
            post_betant = np.array(trace.posterior['beta_nt']).flatten()

            post_p0nt = np.array(trace.posterior['p0_nt']).flatten()

            pnt_pars = np.empty((nsamp, 2))
            pnt_pars[:,0] = post_p0nt
            pnt_pars[:,1] = post_betant

            Mhyd.pnt_pars = pnt_pars

            Mhyd.pnt_model = 'Ettori'

    if not wlonly:
        alldens = np.sqrt(np.dot(Mhyd.Kdens, np.exp(samples.T)) * Mhyd.transf)

        if Mhyd.cf_prof is not None:
            pmc = np.median(alldens, axis=1) / np.sqrt(Mhyd.ccf)
            pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
            pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)

        else:
            pmc = np.median(alldens, axis=1) / np.sqrt(Mhyd.ccf)
            pmcl = np.percentile(alldens, 50. - 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)
            pmch = np.percentile(alldens, 50. + 68.3 / 2., axis=1) / np.sqrt(Mhyd.ccf)

        Mhyd.dens = pmc
        Mhyd.dens_lo = pmcl
        Mhyd.dens_hi = pmch

    if wlonly:
        # Get chains and save them to file
        chain_rd = np.array(trace.posterior['rdelta'])

        nsamp = len(chain_rd.flatten())
    else:
        nsamp = len(samples)

    samppar = np.empty((nsamp, model.npar))
    for i in range(model.npar):

        name = model.parnames[i]
        samppar[:, i] = np.array(trace.posterior[name]).flatten()

    if not wlonly:
        samplogp0 = np.array(trace.posterior['logp0']).flatten()
        Mhyd.samplogp0 = samplogp0
        if fit_eta:
            eta = np.array(trace.posterior['eta']).flatten()
            Mhyd.eta = eta

    Mhyd.samppar = samppar


    Mhyd.elong = elong

    #Mhyd.r3d = samppar.T[1,:] * (elong**(1/3)) #???? who tells that samppar dimension 1 is a distance? What if there is only one parameter?

    if Mhyd.spec_data is not None and not wlonly:
        kt_mod = kt_from_samples(Mhyd, model, nmore=nmore)
        Mhyd.ktmod = kt_mod['TSPEC']
        Mhyd.ktmod_lo = kt_mod['TSPEC_LO']
        Mhyd.ktmod_hi = kt_mod['TSPEC_HI']
        Mhyd.kt3d = kt_mod['T3D']
        Mhyd.kt3d_lo = kt_mod['T3D_LO']
        Mhyd.kt3d_hi = kt_mod['T3D_HI']

    if Mhyd.sz_data is not None and not wlonly:
        pmed, plo, phi = P_from_samples(Mhyd, model, nmore=nmore)
        Mhyd.pmod = pmed
        Mhyd.pmod_lo = plo
        Mhyd.pmod_hi = phi

    totlike = 0.
    nptot = model.npar + 1
    thermolike = 0.
    npthermo = model.npar + 1

    # Mhyd.trace.log_likelihood['tot'] = 0.
    #
    # if fit_bkg:
    #     totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['counts']), axis=2).flatten()
    #     nptot = nptot + npt + 1
    #     #Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['counts']
    #
    # else:
    #     totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['sb']), axis=2).flatten()
    #     nptot = nptot + npt
    #     #Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['sb']
    #
    # if Mhyd.spec_data is not None:
    #     totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['kt']), axis=2).flatten()
    #     thermolike = thermolike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['kt']), axis=2).flatten()
    #     Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['tot'] + Mhyd.trace.log_likelihood['kt']
    #
    #
    # if Mhyd.sz_data is not None:
    #     totlike = totlike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['P']), axis=2).flatten()
    #     thermolike = thermolike + np.sum(np.asarray(Mhyd.trace['log_likelihood']['P']), axis=2).flatten()
    #     Mhyd.trace.log_likelihood['tot'] = Mhyd.trace.log_likelihood['tot'] + Mhyd.trace.log_likelihood['P']
    #
    # if pnt:
    #     nptot = nptot + 3
    #     npthermo = npthermo + 3
    #
    # Mhyd.totlike = totlike
    # Mhyd.nptot = nptot
    # Mhyd.thermolike = thermolike
    # Mhyd.npthermo = npthermo
    # Mhyd.waic = az.waic(Mhyd.trace, var_name='tot')
    # Mhyd.loo = az.loo(Mhyd.trace, var_name='tot')

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
                 max_rad = 0.
                 ):

        logger = logging.getLogger('hydromass')
        if not logger.hasHandlers():
            # Fallback: Define the "default" here because nothing else exists
            logger.setLevel(logging.INFO)
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(logging.Formatter('%(asctime)s @%(name)-25s "%(message)s"', datefmt = '%d-%m-%y %H:%M'))
            logger.addHandler(sh)

        self.logger = create_logger()

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
        self.spec_data = spec_data
        self.sz_data = sz_data

        # Make sure the conversion from arcmin to kpc is done in the correct cosmology, i.e. use amin2kpc just calculated
        for sb in self.sb_data:
            sb.kpc_data(self.amin2kpc)
        for sp in self.spec_data:
            if sp is not None:
                sp.kpc_data(self.amin2kpc)
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
                Z_ab = float(np.mean([np.average(this_spec_data.zfe, weights=1. / this_spec_data.errt_zfe ** 2) for this_spec_data in self.spec_data]))
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


    def fit(self,
            method,
            fit_SB=False, fit_bkg=False, bkglim_amin=None, backval=None,
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
            ):
        '''

        :param method:
            Method used to get the 3-dimensional temperature. One of \' MASSMOD\', \' FORW\', \' GP\'.
            Note: for MASSMOD the total mass is assumed to be: Mtot = Mgas + Mstar + Mother, with Mgas given by the gas mass from the integral of the gas density,
            Mstar given by star mass component, and Mother given by the sum of models_mass
        :param fit_SB:
            Whether to fit the surface brightness profile or not. Default False.
        :param fit_bkg:
            Whether BKG is to be fitted or not. Default False: BKG subtracted and error added in quadrature to SB points.
        :param bkglim_amin:
            Region beyond which estimate the background value. Default last data point.
        :param backval:
            Estimated background value. Default None: compute from bkglim_amin parameter.
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
        '''
        if (fit_SB is False) and (fit_kT is False) and (fit_Z_Ab is False) and (fit_SZ is False) and (fit_WL is False):
            raise ValueError('Nothing to do. At least one of [fit_SB, fit_kT, fit_Z_Ab, fit_SZ, fit_WL] must be set to True')

        if fit_Z_Ab and model_Z_Ab is None:
            raise ValueError('fit_Z_Ab is True but model_Z_Ab is not set')

        dict_utils = common_utils(self, rmin_amin=rmin_amin, rmax_amin=rmax_amin)

        sb_utils(self, dict_utils, sb_setup=sb_setup, fit_bkg=fit_bkg, bkglim_amin=bkglim_amin, backval=backval)

        if fit_kT:
            spec_utils(self, dict_utils)

        # ccf_utils(self, dict_utils)

        if method == 'MASSMOD':
            mass_utils(self, dict_utils, mstar=mstar, p0_prior=p0_prior, p0_type=p0_type)
        elif method == 'FORW':
            forw_utils(self, dict_utils)
        elif method == 'GP':
            gp_utils(self, dict_utils)
        else:
            raise ValueError(f'{method = } not recognized. Use \'MASSMOD\', \'FORW\', or \'GP\'')

        hydro_model = pm.Model()
        with hydro_model:
            params = dict()

            get_density(params, dict_utils, self, fit_bkg=fit_bkg)

            get_metal(params, model_Z_Ab, self, dict_utils)

            if fit_Z_Ab:
                likelihood_metal(params, model_Z_Ab, self, dict_utils)

            if method == 'MASSMOD':
                kt_from_mass(params, models_mass, self, dict_utils, mstar=mstar, model_mstar=model_mstar, model_missing_mass=model_missing_mass)
            elif method == 'FORW':
                kt_from_forw(params, model_P, self, dict_utils)
            elif method == 'GP':
                kt_from_gp(params, models, self, dict_utils)
            else:
                raise ValueError(f'{method = } not recognized. Use \'MASSMOD\', \'FORW\', or \'GP\'')

            if fit_kT:
                likelihood_kt(params, self, dict_utils)

            get_ccf(params, self, dict_utils, self.conversion_factor_type)


            if fit_SB:
                likelihood_sb(params, dict_utils, self, fit_bkg=fit_bkg)

        with hydro_model:
            # start = pm.find_MAP(start=initvals), initvals=start

            trace = pmjax.sample_numpyro_nuts(num_samples, tune=num_warmup, target_accept=target_accept)


















    def run(self, model=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None,
            nrc = None, min_rc = None, max_rc = None,
            nbetas = 6, min_beta = 0.6, max_beta = 3,
            nmore=5,
            p0_prior=None, tune=500, dmonly=False, mstar=None, find_map=True, pnt=False,
            rmin=None, rmax=None, p0_type='sb', init='ADVI', target_accept=0.9,
            pnt_model='Ettori', fit_elong=False, fit_eta=False, use_jax=True, wlonly=False, pnt_prior='sim',
            fix_beta_pnt=False, do_ppc=True, cM_relation=False):
        '''
        Optimize the mass model using the :func:`hydromass.mhyd.Run_Mhyd_PyMC3` function.

        :param model:  A :class:`hydromass.functions.Model` object including the chosen mass model and its input values (mandatory input)
        :type model: class:`hydromass.functions.Model`
        :param bkglim: Limit (in arcmin) out to which the SB data will be fitted; if None then the whole range is considered. Defaults to None.
        :type bkglim: float
        :param nmcmc: Number of PyMC3 steps. Defaults to 1000
        :type nmcmc: int
        :param fit_bkg: Choose whether the counts and the background will be fitted on-the-fly using a Poisson model (fit_bkg=True) or if the surface brightness will be fitted, in which case it is assumed that the background has already been subtracted and Gaussian likelihood will be used (default = False)
        :type fit_bkg: bool
        :param back: Input value for the background. If None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True. Defaults to None.
        :type back: float
        :param samplefile: Name of ASCII file to output the final PyMC3 samples
        :type samplefile: str
        :param nrc: Number of core radii values to set up the multiscale model. Defaults to the number of data points / 4
        :type nrc: int
        :param nbetas: Number of beta values to set up the multiscale model (default = 6)
        :type nbetas: int
        :param min_beta: Minimum beta value (default = 0.6)
        :type min_beta: float
        :param nmore: Number of points to the define the fine grid onto which the mass model and the integration are performed, i.e. for one spectroscopic/SZ value, how many grid points will be defined. Defaults to 5.
        :type nmore: int
        :param p0_prior: Set of two values defining the mean and standard deviation of the Gaussian prior on p0. If None, the code attempts to determine the value of P0 using the :func:`hydromass.plots.estimate_P0` function, which fits a rough gNFW function to estimate the shape of the pressure profile and uses the fitted function to approximate the value of P0.
        :type p0_prior: numpy.ndarray
        :param tune: Number of NUTS tuning steps. Defaults to 500
        :type tune: int
        :param dmonly: Specify whether the mass model is fitted to the total mass (dmonly=False) or to the dark matter only after subtracting the gas mass and the stellar mass if provided (dmonly=True). Defaults to False.
        :type dmonly: bool
        :param mstar: If dmonly=True, provide an array containing the cumulative stellar mass profile, which will be subtracted when adjusting the mass model to the dark matter only.
        :type mstar: numpy.ndarray
        :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
        :type find_map: bool
        :param pnt: Attempt to model the non-thermal pressure profile. If pnt=True, the non-thermal pressure profile of Angelinelli et al. 2020 and the corresponding parameters are used to marginalize over the impact of non-thermal pressure. Defaults to False.
        :type pnt: bool
        :param rmin: Minimum limiting radius (in arcmin) of the active region for the surface brightness. If rmin=None, no minimum radius is applied. Defaults to None.
        :type rmin: float
        :param rmax: Maximum limiting radius (in arcmin) of the active region for the surface brightness. If rmax=None, no maximum radius is applied. Defaults to None.
        :type rmax: float
        :param p0_type: For the estimation of P0, choose whether we will use the surface brightness profile (p0_type='sb') or the spectral normalization (p0_type='norm'). Defaults to 'sb'.
        :type p0_type: str
        :param init: Choose the initialization method for PyMC. Defaults to 'ADVI'
        :type init: str
        :param target_accept: Set the target_accept parameter for PyMC. Defaults to 0.9
        :type target_accept: float
        :param pnt_model: Choose the analytic model for non-thermal pressure modeling. Available choices are 'Angelinelli' (Angelinelli+20) and 'Ettori' (Ettori\&Eckert22). Defaults to 'ettori'
        :type pnt_model: str
        :param fit_elong: Set whether the elongation of the system along the line of sight as a free model parameter to account for differences between various observables induced by line-of-sight elongation. Defaults to False
        :type fit_elong: bool
        :param use_jax: Use JAX optimization when sampling using the numpyro NUTS implementation. Defaults to True
        :type use_jax: bool
        :param wlonly: Set whether the fit will be done to weak lensing data only. Defaults to False
        :type wlonly: bool
        :param pnt_prior: Choose whether informative priors from simulations will be applied to the Pnt profile (pnt_prior='sim'). Alternatively, flat priors will be adopted. Defaults to 'sim'.
        :type pnt_prior: str
        :param fit_eta: Fit for $\\eta_T = P_X/P_{SZ}$ according to the definition of Kozmanyan et al. 2019. Defaults to False.
        :type fit_eta: bool
        :param fix_beta_pnt: When fitting for non-thermal pressure, fix the $\\beta_{NT}$ parameter (i.e. the slope of the polytropic relation) to the typical value of 0.9 (Ettori & Eckert 2022). Defaults to False.
        :type fix_beta_pnt: bool

        '''

        if model is None:

            self.logger.info('Error: No mass model provided')

            return

        Run_Mhyd_PyMC3(self,
                       model=model,
                       bkglim=bkglim,
                       nmcmc=nmcmc,
                       fit_bkg=fit_bkg,
                       back=back,
                       samplefile=samplefile,
                       nrc = nrc, min_rc = min_rc, max_rc = max_rc,
                       nbetas = nbetas, min_beta = min_beta, max_beta = max_beta,
                       nmore=nmore,
                       p0_prior=p0_prior,
                       tune=tune,
                       dmonly=dmonly,
                       mstar=mstar,
                       find_map=find_map,
                       pnt=pnt,
                       pnt_model=pnt_model,
                       rmin=rmin,
                       rmax=rmax,
                       p0_type=p0_type,
                       init=init,
                       target_accept=target_accept,
                       fit_elong=fit_elong,
                       fit_eta=fit_eta,
                       use_jax=use_jax,
                       wlonly=wlonly,
                       pnt_prior=pnt_prior,
                       fix_beta_pnt=fix_beta_pnt,
                       do_ppc=do_ppc,
                       cM_relation=cM_relation)


    def run_forward(self, forward=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None,
                    nrc = None, min_rc = None, max_rc = None,
                    nbetas = 6, min_beta = 0.6, max_beta = 3,
                    nmore=5, tune=500, find_map=True, rmin=0, rmax=None, force_positive_mass = False,
                    force_convective_stability = False, force_increasing_mass = False, fit_eta=False, do_ppc=True):

        '''
        Optimize a parametric forward fit to the gas pressure profile using the :func:`hydromass.forward.Run_Forward_PyMC3` function

        :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object including the loaded data and initial setup (mandatory input)
        :type Mhyd: class:`hydromass.mhyd.Mhyd`
        :param model:  A :class:`hydromass.forward.Forward` object including the definition of the forward model and its input values (mandatory input)
        :type model: class:`hydromass.forward.Forward`
        :param bkglim: Limit (in arcmin) out to which the SB data will be fitted; if None then the whole range is considered. Defaults to None.
        :type bkglim: float
        :param nmcmc: Number of PyMC3 steps. Defaults to 1000
        :type nmcmc: int
        :param fit_bkg: Choose whether the counts and the background will be fitted on-the-fly using a Poisson model (fit_bkg=True) or if the surface brightness will be fitted, in which case it is assumed that the background has already been subtracted and Gaussian likelihood will be used (default = False)
        :type fit_bkg: bool
        :param back: Input value for the background. If None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True. Defaults to None.
        :type back: float
        :param samplefile: Name of ASCII file to output the final PyMC3 samples
        :type samplefile: str
        :param nrc: Number of core radii values to set up the multiscale model. Defaults to the number of data points / 4
        :type nrc: int
        :param nbetas: Number of beta values to set up the multiscale model (default = 6)
        :type nbetas: int
        :param min_beta: Minimum beta value (default = 0.6)
        :type min_beta: float
        :param nmore: Number of points to the define the fine grid onto which the mass model and the integration are performed, i.e. for one spectroscopic/SZ value, how many grid points will be defined. Defaults to 5.
        :type nmore: int
        :param tune: Number of NUTS tuning steps. Defaults to 500
        :type tune: int
        :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
        :type find_map: bool
        :param rmin: Minimum limiting radius (in arcmin) of the active region for the surface brightness. If rmin=None, no minimum radius is applied. Defaults to None.
        :type rmin: float
        :param rmax: Maximum limiting radius (in arcmin) of the active region for the surface brightness. If rmax=None, no maximum radius is applied. Defaults to None.
        :type rmax: float
        '''
        if forward is None:

            self.logger.info('Error no forward model provided')

            return


        Run_Forward_PyMC3(self,
                          Forward=forward,
                          bkglim=bkglim,
                          nmcmc=nmcmc,
                          fit_bkg=fit_bkg,
                          back=back,
                          samplefile=samplefile,
                          nrc = nrc, min_rc = min_rc, max_rc = max_rc,
                          nbetas = nbetas, min_beta = min_beta, max_beta = max_beta,
                          nmore=nmore,
                          tune=tune,
                          find_map=find_map,
                          rmin=rmin,
                          rmax=rmax,
                          fit_eta=fit_eta,
                          do_ppc = do_ppc,
                          force_positive_mass = force_positive_mass,
                          force_convective_stability = force_convective_stability,
                          force_increasing_mass=force_increasing_mass
                          )

    def run_polytropic(self, Polytropic=None, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5, tune=500, find_map=True):
        '''
        Run a polytropic reconstruction with an effective polytropic index model set by the :class:`hydromass.polytropic.Polytropic` class. See :func:`hydromass.polytropic.Run_Polytropic_PyMC3`

        :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object including the loaded data and initial setup (mandatory input)
        :type Mhyd: class:`hydromass.mhyd.Mhyd`
        :param Polytropic: Polytropic model defined using the :class:`hydromass.polytropic.Polytropic` class
        :type Polytropic: :class:`hydromass.polytropic.Polytropic`
        :param bkglim: Limit (in arcmin) out to which the SB data will be fitted; if None then the whole range is considered. Defaults to None.
        :type bkglim: float
        :param nmcmc: Number of PyMC3 steps. Defaults to 1000
        :type nmcmc: int
        :param fit_bkg: Choose whether the counts and the background will be fitted on-the-fly using a Poisson model (fit_bkg=True) or if the surface brightness will be fitted, in which case it is assumed that the background has already been subtracted and Gaussian likelihood will be used (default = False)
        :type fit_bkg: bool
        :param back: Input value for the background. If None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True. Defaults to None.
        :type back: float
        :param samplefile: Name of ASCII file to output the final PyMC3 samples
        :type samplefile: str
        :param nrc: Number of core radii values to set up the multiscale model. Defaults to the number of data points / 4
        :type nrc: int
        :param nbetas: Number of beta values to set up the multiscale model (default = 6)
        :type nbetas: int
        :param min_beta: Minimum beta value (default = 0.6)
        :type min_beta: float
        :param nmore: Number of points to the define the fine grid onto which the mass model and the integration are performed, i.e. for one spectroscopic/SZ value, how many grid points will be defined. Defaults to 5.
        :type nmore: int
        :param tune: Number of NUTS tuning steps. Defaults to 500
        :type tune: int
        :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
        :type find_map: bool
        '''
        if Polytropic is None:

            self.logger.info('Error no polytropic model provided')

            return


        Run_Polytropic_PyMC3(self,
                          Polytropic=Polytropic,
                          bkglim=bkglim,
                          nmcmc=nmcmc,
                          fit_bkg=fit_bkg,
                          back=back,
                          samplefile=samplefile,
                          nrc=nrc,
                          nbetas=nbetas,
                          min_beta=min_beta,
                          nmore=nmore,
                          tune=tune,
                          find_map=find_map)


    def run_GP(self, bkglim=None, nmcmc=1000, fit_bkg=False, back=None,
            samplefile=None,
               nrc = None, min_rc = None, max_rc = None,
               nbetas = 6, min_beta = 0.6, max_beta = 3
               , nmore=5, tune=500, find_map=True,
            bin_fact=1.0, smin=None, smax=None, ngauss=100, extend=False, T0extend=None, rmin=None, rmax=None, force_positive_mass = False,
               force_convective_stability = False, force_increasing_mass=False):

        '''
        Run a non-parametric log-normal mixture reconstruction. See :func:`hydromass.nonparametric.Run_NonParametric_PyMC3`

        :param bkglim: Radius (in arcmin) beyond which it is assumed that the background fully dominates the profile. If None, the entire radial range is fitted as source + background. Defaults to None.
        :type bkglim: float
        :param nmcmc: Number of NUTS samples. Defaults to 1000
        :type nmcmc: int
        :param fit_bkg: Define whether we will fit the counts as source + background (True) or the background subtracted surface brightness as source only (False). Defaults to False.
        :type fit_bkg: bool
        :param back: Input value for the background. If None then the mean surface brightness in the region outside "bkglim" is used. Relevant only if fit_bkg = True. Defaults to None.
        :type back: float
        :param samplefile: Name of ASCII file to output the final PyMC3 samples
        :type samplefile: str
        :param nrc: Number of core radii values to set up the multiscale model. Defaults to the number of data points / 4
        :type nrc: int
        :param nbetas: Number of beta values to set up the multiscale model (default = 6)
        :type nbetas: int
        :param min_beta: Minimum beta value (default = 0.6)
        :type min_beta: float
        :param nmore: Number of points to the define the fine grid onto which the mass model and the integration are performed, i.e. for one spectroscopic/SZ value, how many grid points will be defined. Defaults to 5.
        :type nmore: int
        :param tune: Number of NUTS tuning steps. Defaults to 500
        :type tune: int
        :param bin_fact: bin_fact: Binning factor for the definition of the log-normal standard deviations, i.e. the standard deviations of the log-normals will be set to bin_fact * (rout - rin). The larger the value of bin_fact the stronger the smoothing, but the less accurate and flexible the model. Defaults to 1.
        :type bin_fact: float
        :param smin: Minimum value of the log-normal standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
        :type smin: float
        :param smax: Maximum value of the log-normal standard deviation. If None, the width of the bins will be used (see bin_fact). If the value is set, the smoothing scales are set as logarithmically spaced between smin and smax. Defaults to None.
        :type smax: float
        :param ngauss: Number of log-normal functions. Defaults to 100
        :type ngauss: int
        :param find_map: Specify whether a maximum likelihood fit will be performed first to initiate the sampler. Defaults to True
        :type find_map: bool
        '''

        Run_NonParametric_PyMC3(self,
                                bkglim=bkglim,
                                nmcmc=nmcmc,
                                fit_bkg=fit_bkg,
                                back=back,
                                samplefile=samplefile,
                                nrc = nrc, min_rc = min_rc, max_rc = max_rc,
                                nbetas = nbetas, min_beta = min_beta, max_beta = max_beta,
                                nmore=nmore,
                                tune=tune,
                                find_map=find_map,
                                bin_fact=bin_fact,
                                smin=smin,
                                smax=smax,
                                ngauss=ngauss,
                                extend=extend,
                                T0extend=T0extend,
                                rmin = rmin,
                                rmax = rmax,
                                force_positive_mass = force_positive_mass,
                                force_convective_stability = force_convective_stability,
                                force_increasing_mass=force_increasing_mass
                                )

    def SaveModel(self, model, outfile=None):
        '''
        Save the output of a mass model fit into a FITS file through the :func:`hydromass.save.SaveModel` function

        :param model: :class:`hydromass.functions.Model` defining the chosen mass model
        :type model: :class:`hydromass.functions.Model`
        :param outfile: Name of output FITS file. If None, the file is outputted to a file called "output_model.fits" under the default output directory specified in the current object. Defaults to None
        :type outfile: str
        '''

        SaveModel(self,
                  model,
                  outfile)

    def SaveGP(self, outfile=None):
        '''
        Save the output of a non-parametric reconstruction into a FITS file through the :func:`hydromass.save.SaveGP` function

        :param outfile: Name of output FITS file. If None, the file is outputted to a file called "output_GP.fits" under the default output directory specified in the current object. Defaults to None
        :type outfile: str
        '''

        SaveGP(self,
               outfile)

    def SaveForward(self, Forward, outfile=None):
        '''
        Save the output of a parametric forward reconstruction into a FITS file through the :func:`hydromass.save.SaveForward` function

        :param Forward: A :class:`hydromass.forward.Forward` object containing the definition of the forward model
        :type Forward: class:`hydromass.forward.Forward`
        :param outfile: Name of output FITS file. If None, the file is outputted to a file called "output_forward.fits" under the default output directory specified in the current object. Defaults to None
        :type outfile: str
        '''

        SaveForward(self,
                    Forward,
                    outfile)
