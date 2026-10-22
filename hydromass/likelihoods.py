import pymc as pm
import pytensor.tensor as tt
import numpy as np

from .constants import cgsamu, cgskpc, Msun, y_prefactor, kev2erg, cgsMpc, cgsG, year
from .utility import mgas_pm
from .deproject import MyDeprojVol

__all__ = ['get_density', 'likelihood_sb', 'get_metal', 'likelihood_metal', 'kt_from_mass', 'likelihood_kt', 'likelihood_sz', 'get_ccf']


def get_density(params, dict_utils, Mhyd, fit_bkg = False):
    testval = dict_utils['testval']
    testbkg = dict_utils['testbkg']
    npt = dict_utils['npt']
    Kdens_m = dict_utils['Kdens_m']
    Kdens_m_out = dict_utils['Kdens_m_out']
    K_list = dict_utils['K_list']

    Mhyd.logger.info('DENSITY PARAMETERS')
    Mhyd.logger.info(f'  pm.Normal(\'coefs\', mu = {testval}, sigma = 20, shape = {npt})')
    coefs = pm.Normal('coefs', mu = testval, sigma = 20, shape = npt)

    if fit_bkg:
        bkgd = pm.Normal('bkg', mu = testbkg, sigma = 0.05, shape = 1)  # in case fit_bkg = False this is not fitted
        ctot = pm.math.concatenate((coefs, bkgd), axis = 0)
        al = pm.math.exp(ctot)
    else:
        al = pm.math.exp(coefs)


    dens_m = pm.math.sqrt(pm.math.dot(Kdens_m, al) * Mhyd.transf)  # electron density in cm-3
    dens_m_out = pm.math.sqrt(pm.math.dot(Kdens_m_out, al) * Mhyd.transf)  # electron density in cm-3

    for i, K in enumerate(K_list):
        if fit_bkg:
            pred = pm.math.dot(K, al) + Mhyd.sb_data.bkgcounts  # Predicted number of counts per annulus
        else:
            pred = pm.math.dot(K, al)
        params[f'pred_{i}'] = pm.Deterministic(f'pred_{i}', pred)

    params['ne_model_rref'] = pm.Deterministic('ne_model_rref', dens_m)
    params['ne_model_rout'] = pm.Deterministic('ne_model_rout', dens_m_out)


def likelihood_sb(params, dict_utils, Mhyd, fit_bkg = False):
    for i, (this_sb_data, valid_sb) in enumerate(zip(Mhyd.sb_data, dict_utils['valid_sb_list'])):
        if fit_bkg:
            countsmod = params[f'pred_{i}'] * params[f'ccf_{i}']
            count_obs = pm.Poisson('counts', mu = countsmod[valid_sb], observed = this_sb_data.counts[valid_sb])  # counts likelihood
            pm.Deterministic('cts_model', countsmod)
        else:
            sbmod = params[f'pred_{i}'] * params[f'ccf_{i}']
            pm.Deterministic(f'sb_model_{i}', sbmod)
            sb_obs = pm.Normal(f'sb_obs_{i}', mu = sbmod[valid_sb], observed = this_sb_data.sb[valid_sb], sigma = this_sb_data.esb[valid_sb])  # Sx likelihood


def get_metal(params, model, Mhyd, dict_utils):
    rref_m = dict_utils['rref_m']
    Mhyd.logger.info('METAL ABUNDANCE PARAMETERS')
    pmod = pm.math.stack(model.pm_priors(Mhyd.logger), axis = 0)

    zfe_model = model.func_pm(rref_m, *pmod)

    zfe_model = tt.clip(zfe_model, 0.01, 1.00)

    params['zfe_model'] = pm.Deterministic('zfe_model', zfe_model)


def likelihood_metal(params, model, Mhyd, dict_utils):
    index_x_list = dict_utils['index_kt_list']
    valid_spec_list = dict_utils['valid_spec_list']

    zfe_model = params['zfe_model']

    for i, (this_spec_data, index_x, valid_spec) in enumerate(zip(Mhyd.spec_data, index_x_list, valid_spec_list)):
        Z_Fe_obs = pm.TruncatedNormal(f'Z_Fe_obs_{i}', mu = zfe_model[index_x][valid_spec], sigma = this_spec_data.errt_zfe[valid_spec], observed = this_spec_data.zfe[valid_spec])



def kt_from_forw(params, model_P, Mhyd, dict_mass_utils, dict_common_utils, dict_spec_utils, dict_sb_utils):
    pass


def kt_from_gp(params, model_gp, Mhyd, dict_mass_utils, dict_common_utils, dict_spec_utils, dict_sb_utils):
    pass


def kt_from_mass(params, models, Mhyd, dict_utils,
                 model_mstar = None, mstar = None, model_missing_mass = None):
    # Mstar logic:
    # if mstar=None and model_mstar=None: mass_star=0
    # if mstar=something and model_mstar=None: mass_star=mstar
    # if mstar=None and model_mstar=something: mass_star=model_mstar
    # if mstar=something and model_mstar=something: mass_star=model_mstar ; model_mstar fitted to mstar

    rref_m = dict_utils['rref_m']
    rout_m = dict_utils['rout_m']
    rin_m = dict_utils['rin_m']

    if model_mstar is not None:
        Mhyd.logger.info('STAR MASS PARAMETERS')
        pmod = pm.math.stack(model_mstar.pm_priors(Mhyd.logger), axis = 0)

        mstar_model_rref = model_mstar.func_pm(rref_m, *pmod)

        if mstar is not None:
            mstar_m_rref = dict_utils['mstar_m_rref']
            pm.Normal('mstar_obs', mu = pm.math.log(mstar_model_rref), sigma = 0.1, observed = mstar_m_rref)

        mstar_rref = mstar_model_rref
    elif mstar is not None:
        mstar_rref = dict_utils['mstar_m_rref']
    else:
        mstar_rref = pm.math.constant(0)

    if model_mstar is not None or mstar is not None:
        pm.Deterministic('mstar_model_rref', mstar_rref * Mhyd.mfact0 * 1e13)

    if model_missing_mass is not None:
        Mhyd.logger.info('MISSING MASS PARAMETERS')
        pmod = pm.math.stack(model_missing_mass.pm_priors(Mhyd.logger), axis = 0)

        m_missing_mass_model_rref = model_missing_mass.func_pm(rref_m, *pmod)
        pm.Deterministic('m_missing_mass_model_rref', m_missing_mass_model_rref * Mhyd.mfact0 * 1e13)
    else:
        m_missing_mass_model_rref = pm.math.constant(0)

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3
    mgas_out = mgas_pm(rin_m, rout_m, params['ne_model_rref']) * nhconv / Mhyd.mfact0
    # Assuming mgas(rref) = (mgas(rin)+mgas(rout)) / 2
    mgas_rref = tt.concatenate([[mgas_out[0]], (mgas_out[1:] + mgas_out[:-1])]) / 2
    pm.Deterministic('mgas_model_rref', mgas_rref * Mhyd.mfact0 * 1e13)

    mbar_rref = mgas_rref + mstar_rref + m_missing_mass_model_rref

    pmods = []
    for mass_model in models:
        Mhyd.logger.info(f'MASS MODEL {mass_model.massmod} PARAMETERS')
        pmods.append(pm.math.stack(mass_model.pm_priors(Mhyd.logger), axis = 0))

    # if len(models) > 1:
    #     Mhyd.logger.info(f'MIXTURE MASS PARAMETERS')
    #     # Do mixture model
    #     sum_frac = 0
    #     frac_mass = []
    #     for i in range(len(models) - 1):
    #         Mhyd.logger.info(f'  pm.Uniform(\'f_{i}\', lower=0, upper=1)')
    #         frac = pm.Uniform(f'f_{i}', lower = 0, upper = 1)
    #         sum_frac += frac
    #         frac_mass.append(frac)
    #
    #     # Enforce sum_frac <= 1 (makes last fraction non-negative)
    #     pm.Potential('frac_sum_constraint', tt.switch(sum_frac <= 1, 0.0, -np.inf))
    #     frac_mass.append(1 - sum_frac)
    # else:
    #     frac_mass = [1]

    mass_rref = 0
    mass_rout = 0
    for mass_model, pmod in zip(models, pmods):
        if mass_model.massmod in ['RAR', 'GMOND', 'EMOND']:
            # MOND also needs baryon mass to get phantom mass
            tmass_rref = mass_model.func_pm(rref_m, *pmod, mbar = mbar_rref * 1e13 * Mhyd.mfact0) / Mhyd.mfact0 / 1e13
            tmass_rout = mass_model.func_pm(rout_m, *pmod, mbar = mbar_rref * 1e13 * Mhyd.mfact0) / Mhyd.mfact0 / 1e13
        else:
            tmass_rref = Mhyd.mfact * mass_model.func_pm(rref_m, *pmod, delta = mass_model.delta) / Mhyd.mfact0
            tmass_rout = Mhyd.mfact * mass_model.func_pm(rout_m, *pmod, delta = mass_model.delta) / Mhyd.mfact0

        mass_rref = mass_rref + tmass_rref
        mass_rout = mass_rout + tmass_rout

        pm.Deterministic(f'm{mass_model.massmod.lower()}_model_rref', tmass_rref * Mhyd.mfact0 * 1e13)

    mass_rref += mbar_rref

    pm.Deterministic(f'mtot_model_rref', mass_rref * Mhyd.mfact0 * 1e13)

    # Pressure gradient
    dpres = - mass_rref / rref_m ** 2 * params['ne_model_rref'] * (rout_m - rin_m)

    P0_est = dict_utils['P0_est']
    err_P0_est = dict_utils['err_P0_est']
    Mhyd.logger.info(f'logP0 PARAMETER')
    Mhyd.logger.info(f"  pm.TruncatedNormal('logp0', mu={np.log(P0_est)}, sigma={err_P0_est / P0_est}, "  # +
                     f"lower={np.log(P0_est) - 2 * np.log(10)}, upper={np.log(P0_est) + 2 * np.log(10)})")
    logp0 = pm.TruncatedNormal('logp0', mu = np.log(P0_est), sigma = err_P0_est / P0_est,
                               lower = np.log(P0_est) - 2 * np.log(10),
                               upper = np.log(P0_est) + 2 * np.log(10))

    press00 = pm.math.exp(logp0)

    # directly returns press_out
    press_rout = press00 - pm.math.dot(dict_utils['int_mat'], dpres)

    # MassModel temperature
    t3d_out = press_rout / params['ne_model_rout']
    # Mazzotta weights
    ei = params['ne_model_rout'] ** 2 * t3d_out ** (-0.75)

    params['t3d_out'] = pm.Deterministic('t3d_out', t3d_out)
    params['press_rout'] = pm.Deterministic('press_rout', press_rout)

    for i, (this_sb_data, proj_mat_kt, proj_mat_sb) in enumerate(zip(Mhyd.sb_data, dict_utils['proj_mat_kt_list'], dict_utils['proj_mat_sb_list'])):
        # Temperature projection
        flux = pm.math.dot(proj_mat_kt, ei)
        #
        tproj = pm.math.dot(proj_mat_kt, t3d_out * ei) / flux
        #
        params[f'tproj_{i}'] = pm.Deterministic(f'tproj_{i}', tproj)

        flux_sb = pm.math.dot(proj_mat_sb, ei)
        #
        tproj_sb = pm.math.dot(proj_mat_sb, t3d_out * ei) / flux_sb

        log_tproj_sb = tt.clip(pm.math.log(tproj_sb), this_sb_data.cf_fit.log_kt_min, this_sb_data.cf_fit.log_kt_max)  # pm.math.log(tproj_sb)  #

        params[f'log_tproj_sb_{i}'] = pm.Deterministic(f'log_tproj_sb_{i}', log_tproj_sb)


def likelihood_kt(params, Mhyd, dict_utils):

    for i, (this_spec_data, valid_spec) in enumerate(zip(Mhyd.spec_data, dict_utils['valid_spec_list'])):

        tproj = params[f'tproj_{i}']

        T_obs = pm.Normal(f'T_obs_{i}', mu = tproj[valid_spec], observed = this_spec_data.temp_x[valid_spec], sigma = this_spec_data.errt_x[valid_spec])  # temperature likelihood


def likelihood_sz(params, Mhyd, dict_utils, fit_eta = False, nmore = 5):
    '''
    CLAUDE: SZ likelihood for each dataset in Mhyd.sz_data, from the model pressure params['press_rout'] (see :func:`kt_from_mass`).
    Depending on sz_data.sz_fit, it fits the pressure ('P', 'logP') or the Compton y profile ('Y', 'logY'). Adapted from :func:`hydromass.mhyd.Run_Mhyd_PyMC3`.

    :param fit_eta: Fit for eta = P_SZ / P_X, with a pm.Uniform('eta', lower=0.2, upper=5) prior. Default False.
    :param nmore: nmore used to build the radial grid; the pressure is extrapolated over 2 * nmore extra shells out to 10 Mpc for the y projection. Default 5.
    '''
    rin_m = dict_utils['rin_m']
    rout_m = dict_utils['rout_m']
    ntm = dict_utils['ntm']

    press_rout = params['press_rout']

    if fit_eta:
        Mhyd.logger.info('ETA PARAMETER')
        Mhyd.logger.info("  pm.Uniform('eta', lower=0.2, upper=5)")
        eta = pm.Uniform('eta', lower = 0.2, upper = 5)
    else:
        eta = 1

    for i, (this_sz_data, index_sz) in enumerate(zip(Mhyd.sz_data, dict_utils['index_sz_list'])):

        if this_sz_data.sz_fit in ['P', 'logP']:  # Fitting the pressure

            Mhyd.logger.info(f'Pressure fit {this_sz_data.sz_fit}')

            pfit = press_rout[index_sz] * eta  # P_SZ = P_X * eta

            params[f'pfit_{i}'] = pm.Deterministic(f'pfit_{i}', pfit)

            if this_sz_data.sz_fit == 'P':
                P_obs = pm.MvNormal(f'P_obs_{i}', mu = pfit, observed = this_sz_data.pres_sz, cov = this_sz_data.covmat_sz)  # SZ pressure likelihood
            else:
                P_obs = pm.MvNormal(f'P_obs_{i}', mu = pm.math.log(pfit), observed = this_sz_data.logp_sz, cov = this_sz_data.covmat_sz)  # SZ pressure likelihood

        if this_sz_data.sz_fit in ['Y', 'logY']:

            Mhyd.logger.info(f'Compton y fit {this_sz_data.sz_fit}')

            nout = 2 * nmore

            rout_m_p = np.append(rout_m, np.logspace(np.log10(np.max(rout_m) * 1.1), np.log10(10000.), nout))
            rin_m_p = np.append(rin_m, rout_m_p[ntm - 1:ntm - 1 + nout])

            # Power-law extrapolation of the pressure beyond the grid, using the slope of the last nout points
            slope = (pm.math.log(press_rout[ntm - 1]) - pm.math.log(press_rout[ntm - nout])) / (
                    np.log(rout_m[ntm - 1]) - np.log(rout_m[ntm - nout]))

            press_out = press_rout[ntm - 1] * (rout_m_p[ntm:] / rout_m[ntm - 1]) ** slope

            press_p = pm.math.concatenate([press_rout, press_out], axis = 0)

            proj_vol = MyDeprojVol(rin_m_p, rout_m_p).deproj_vol().T

            area_proj = np.pi * (-(rin_m_p) ** 2 + (rout_m_p) ** 2)

            integ = pm.math.dot(proj_vol, press_p) / area_proj * cgskpc

            y_num = y_prefactor * integ  # prefactor in cm2/keV

            yfit = y_num[index_sz] * eta

            if this_sz_data.psfmat is not None:
                yfit = pm.math.dot(this_sz_data.psfmat, yfit)

            params[f'yfit_{i}'] = pm.Deterministic(f'yfit_{i}', yfit)

            if this_sz_data.sz_fit == 'Y':
                Y_obs = pm.MvNormal(f'Y_obs_{i}', mu = yfit, observed = this_sz_data.y_sz, cov = this_sz_data.covmat_sz)
            else:
                Y_obs = pm.MvNormal(f'Y_obs_{i}', mu = pm.math.log(yfit), observed = this_sz_data.logy_sz, cov = this_sz_data.covmat_sz)


def get_ccf(params, Mhyd, dict_utils, conversion_factor_type):
    if conversion_factor_type == 'table':
        zfe_model = params['zfe_model']
        for i, (this_sb_data, index_sb) in enumerate(zip(Mhyd.sb_data, dict_utils['index_sb_list'])):
            ccf = this_sb_data.cf_fit.predict_cf_pm(params[f'log_tproj_sb_{i}'], zfe_model[index_sb])

            params[f'ccf_{i}'] = pm.Deterministic(f'ccf_{i}', ccf)
    else:
        for i, this_sb_data in enumerate(Mhyd.sb_data):
            params[f'ccf_{i}'] = this_sb_data.ccf
