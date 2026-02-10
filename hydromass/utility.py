import numpy as np


from .deproject import list_params, calc_linear_operator, calc_sb_operator, list_params_density, calc_density_operator, MyDeprojVol
from .plots import rads_more



# This code used to be repeated many times in many different places. Now it is in one place and edits to sb deprojection should be made here
def sb_utils(Mhyd, fit_bkg = False, rmin=None, rmax=None, bkglim=None, back=None, nrc=None, nbetas=6, min_beta=0.6, nmore=5):
    prof = Mhyd.sbprof
    sb = prof.profile.astype('float32')
    esb = prof.eprof.astype('float32')
    rad = prof.bins.astype('float32')
    erad = prof.ebins.astype('float32')

    bkgcounts, counts = None, None
    if fit_bkg:
        if prof.counts is not None:
            counts = prof.counts.astype('int32')
            bkgcounts = prof.bkgcounts.astype('float32')
        else:
            Mhyd.logger.info('The fit_bkg option can only be used when fitting counts, which are not available. Reverting to default')
            fit_bkg = False

    if not prof.voronoi:
        area = prof.area.astype('float32')
        exposure = prof.effexp.astype('float32')

    if rmax is None:
        rmax = np.max(rad+erad)

    if rmin is None:
        rmin = 0

    valid = np.where(np.logical_and(rad >= rmin, rad < rmax))

    # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    if bkglim is None:
        bkglim=float(np.max(rad+erad))
        if back is None:
            back = sb[len(sb) - 1]
    else:
        backreg = np.where(rad>bkglim)
        if back is None:
            back = np.mean(sb[backreg])

    # Set source region
    sourcereg = np.where(rad < bkglim)

    # Set vector with list of parameters
    pars = list_params(rad, sourcereg, nrc, nbetas, min_beta)


    npt = len(pars)

    if prof.psfmat is not None:
        psfmat = prof.psfmat
    else:
        psfmat = np.eye(len(sb))

    # Compute linear combination kernel
    if fit_bkg:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg = True)

        K = calc_linear_operator(rad, sourcereg, pars, area, exposure, psfmat) # transformation to counts

    else:

        Ksb = calc_sb_operator(rad, sourcereg, pars, withbkg=False)

        K = np.dot(psfmat, Ksb)

    # Set up initial values
    if np.isnan(sb[0]) or sb[0] <= 0:
        testval = -10.
    else:
        testval = np.log(sb[0] / npt)
    if np.isnan(back) or back <= 0 or back is None:
        testbkg = -10.
    else:
        testbkg = np.log(back)

    pardens = list_params_density(rad, sourcereg, Mhyd.amin2kpc, nrc, nbetas, min_beta)

    if fit_bkg:

        Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc)

    else:

        Kdens = calc_density_operator(rad, pardens, Mhyd.amin2kpc, withbkg=False)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore = nmore)

    rref_m = (rin_m + rout_m) / 2.

    vx = MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc)

    vol = vx.deproj_vol().T

    Mhyd.cf_prof = None

    try:
        nn = len(Mhyd.ccf)

    except TypeError:

        Mhyd.logger.info('Single conversion factor provided, we will assume it is constant throughout the radial range')

        cf = Mhyd.ccf

    else:

        if len(Mhyd.ccf) != len(rad):

            Mhyd.logger.info('The provided conversion factor has a different length as the input radial binning. Adopting the mean value.')

            cf = np.mean(Mhyd.ccf)

        else:

            Mhyd.logger.info('Interpolating conversion factor profile onto the radial grid')

            cf = np.interp(rout_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

            Mhyd.cf_prof = cf

    proj_mat = None

    if Mhyd.spec_data is not None:

        if Mhyd.spec_data.psfmat is not None:

            mat1 = np.dot(Mhyd.spec_data.psfmat.T, sum_mat)

            proj_mat = np.dot(mat1, vol)

        else:

            proj_mat = np.dot(sum_mat, vol)

    if fit_bkg:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)


    Mhyd.bkglim = bkglim

    Mhyd.nrc = nrc

    Mhyd.nbetas = nbetas

    Mhyd.min_beta = min_beta

    Mhyd.fit_bkg = fit_bkg

    Mhyd.nmore = nmore





    Mhyd.pars = pars

    Mhyd.sourcereg = sourcereg

    Mhyd.pardens = pardens

    Mhyd.K = K

    Mhyd.Kdens = Kdens

    Mhyd.Ksb = Ksb

    Mhyd.Kdens_m = Kdens_m









    nbin = len(sb)

    return (testval, testbkg, npt, bkgcounts, counts, sb, esb, valid, cf, rin_m, rout_m, rref_m, proj_mat, index_sz, ntm, rad,
            rmin, rmax, vol, nbin)


















