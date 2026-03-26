import numpy as np


from .deproject import list_params, calc_linear_operator, calc_sb_operator, list_params_density, calc_density_operator, MyDeprojVol, calc_grad_operator
from .constants import cgskpc, cgsamu, Msun

__all__ = ['rads_more', 'dens_utils', 'sb_utils']

def rads_more(Mhyd, nmore=5, extend=False):
    """

    Return grid of (in, out) radii from X-ray, SZ data or both. Concatenates radii if necessary, then computes a grid of radii.
    Returns the output arrays and the indices corresponding to the input X-ray and/or SZ radii.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing loaded X-ray and/or SZ loaded data.
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
    :type nmore: int
    :return:
        - rin, rout: the inner and outer radii of the fine grid
        - index_x, index_sz: lists of indices corresponding to the position of the input values in the grid
        - sum_mat: matrix containing the number of values in each subgrid bin
        - ntm: total number of grid points
    """
    if Mhyd.spec_data is not None and Mhyd.sz_data is None:

        rout_joint = Mhyd.spec_data.rout_x

        rref_joint = Mhyd.spec_data.rref_x

    elif Mhyd.spec_data is None and Mhyd.sz_data is not None:

        rout_joint = Mhyd.sz_data.rout_sz

        rref_joint = Mhyd.sz_data.rref_sz

    elif Mhyd.spec_data is not None and Mhyd.sz_data is not None:

        rout_joint = np.sort(np.append(Mhyd.spec_data.rout_x, Mhyd.sz_data.rout_sz))

        rref_joint = np.sort(np.append(Mhyd.spec_data.rref_x, Mhyd.sz_data.rref_sz))

    else:

        print('No loaded data found in input hydromass.Mhyd object, nothing to do')

        return

    rin_joint = np.roll(rout_joint, 1)

    rin_joint[0] = 0.

    njoint = len(rref_joint)

    tot_joint = np.sort(np.append(rin_joint, rref_joint))

    ntotjoint = len(tot_joint)

    ntm = int((ntotjoint - 0.5) * nmore)

    rout_more = np.empty(ntm)

    for i in range(ntotjoint - 1):

        rout_more[i * nmore:(i + 1) * nmore] = np.linspace(tot_joint[i], tot_joint[i + 1], nmore + 1)[1:]

    rout_more[(ntotjoint - 1) * nmore:] = np.linspace(rref_joint[njoint - 1], rout_joint[njoint - 1], int(nmore / 2.) + 1)[1:]

    # Move the outer boundary to the edge of the SB profile if it is farther out
    sbprof = Mhyd.sbprof

    if Mhyd.max_rad > np.max(rout_more):
        nvm = len(rout_more)

        dx_out = rout_more[nvm - 1] - rout_more[nvm - 2]

        rout_2add = np.arange(np.max(rout_more), Mhyd.max_rad, dx_out)

        rout_2add = np.append(rout_2add[1:], Mhyd.max_rad)

        rout_more = np.append(rout_more, rout_2add)

        ntm = len(rout_more)

    rin_more = np.roll(rout_more, 1)

    rin_more[0] = 0

    index_x, index_sz = None, None

    if Mhyd.spec_data is not None:

        index_x = np.where(np.in1d(rout_more, Mhyd.spec_data.rref_x))

    if Mhyd.sz_data is not None:

        index_sz = np.where(np.in1d(rout_more, Mhyd.sz_data.rref_sz))

    sum_mat = None

    if Mhyd.spec_data is not None:

        ntot = len(rout_more)

        nx = len(Mhyd.spec_data.rref_x)

        if not extend:

            sum_mat = np.zeros((nx, ntot))

        else:

            if Mhyd.spec_data.rout_x[nx-1] < np.max(rin_more):
                sum_mat = np.zeros((nx+1, ntot))

            else:
                sum_mat = np.zeros((nx, ntot))

        for i in range(nx):

            ix = np.where(np.logical_and(rin_more < Mhyd.spec_data.rout_x[i], rin_more >= Mhyd.spec_data.rin_x[i]))

            nval = len(ix[0])

            sum_mat[i, :][ix] = 1. / nval

        if extend:

            if Mhyd.spec_data.rout_x[nx-1] < np.max(rin_more):

                ix = np.where(rin_more >= Mhyd.spec_data.rout_x[nx-1])

                nval = len(ix[0])

                sum_mat[nx, :][ix] = 1. / nval

    return rin_more, rout_more, index_x, index_sz, sum_mat, ntm

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

    volmat = vx.deproj_vol().T

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

            cf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

            Mhyd.cf_prof = cf

    proj_mat = None

    if Mhyd.spec_data is not None:

        if Mhyd.spec_data.psfmat is not None:

            psfmat = Mhyd.spec_data.psfmat

        else:

            psfmat = np.eye(len(Mhyd.spec_data.temp_x))

        proj_mat = np.dot(np.dot(psfmat, sum_mat), volmat)

    if fit_bkg:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg=False)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, pardens, Mhyd.amin2kpc, withbkg = False)


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

    Mhyd.Kdens_grad = Kdens_grad








    nbin = len(sb)



    return (testval, testbkg, npt, bkgcounts, counts, sb, esb, valid, cf, rin_m, rout_m, rref_m, proj_mat, volmat, index_sz, ntm, rad,
            rmin, rmax, nbin)




def dens_utils(Mhyd, rin=None, rout=None, npt=200, nmore=5):
    '''
    Computes density and gas mass profiles from samples. This method is common to Mhyd, Forw, and NP.

    :param Mhyd:
    :param rin:
    :param rout:
    :param npt:
    :param nmore:
    :return:
    '''
    nsamp = len(Mhyd.samples)

    rin_m, rout_m, index_x, index_sz, sum_mat, ntm = rads_more(Mhyd, nmore=nmore)

    if rin is None:
        rin = np.min((rin_m+rout_m)/2.)

        if rin == 0:
            rin = 1.

    if rout is None:
        rout = np.max(rout_m)

    bins = np.logspace(np.log10(rin), np.log10(rout), npt + 1)

    bins = np.linspace(np.sqrt(rin), np.sqrt(rout), npt + 1)**2

    if rin == 1.:
        bins[0] = 0.

    rin_m = bins[:npt]

    rout_m = bins[1:]

    rref_m = (rin_m + rout_m) / 2.

    nvalm = len(rin_m)

    if Mhyd.cf_prof is not None:

        rad = Mhyd.sbprof.bins

        tcf = np.interp(rref_m, rad * Mhyd.amin2kpc, Mhyd.ccf)

        cf_prof = np.repeat(tcf, nsamp).reshape(nvalm, nsamp)

    else:

        cf_prof = Mhyd.ccf

    if Mhyd.fit_bkg:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc)

    else:

        Kdens_m = calc_density_operator(rref_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg = False)

        Kdens_grad = calc_grad_operator(rout_m / Mhyd.amin2kpc, Mhyd.pardens, Mhyd.amin2kpc, withbkg = False)

    dens_m = np.sqrt(np.dot(Kdens_m, np.exp(Mhyd.samples.T)) / cf_prof * Mhyd.transf)

    # Matrix containing integration volumes
    volmat = np.repeat(4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3), nsamp).reshape(nvalm, nsamp)

    # Compute Mgas profile as cumulative sum over the volume

    nhconv = cgsamu * Mhyd.mu_e * cgskpc ** 3 / Msun  # Msun/kpc^3

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas = np.dot(cs_mat, dens_m * nhconv * volmat)

    return bins, rin_m, rout_m, rref_m, dens_m, mgas, nvalm, nsamp, Kdens_grad, cf_prof













