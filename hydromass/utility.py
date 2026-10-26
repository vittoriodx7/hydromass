from curses import has_key

import ipdb
import numpy as np
import copy
import pymc as pm
import pyproffit
from scipy.special import gamma
from scipy.optimize import minimize

# from .deproject import list_params, calc_linear_operator, calc_sb_operator, calc_density_operator, MyDeprojVol, calc_grad_operator
from .deproject import MyDeprojVol, calc_linear_operator, calc_density_operator, calc_grad_operator

__all__ = ['rads_more', 'list_params', 'sb_utils', 'spec_utils', 'common_utils', 'mass_utils', 'mgas_pm', 'sum_mat_sz']

# Function to create the list of parameters for the basis functions
nsh = 4.  # number of basis functions to set


def mgas_pm(rin_m, rout_m, dens):
    '''
    Theano function to compute the gas mass

    :param rin_m: 1-D array containing the inner edges of radial bins
    :type rin_m: numpy.ndarray
    :param rout_m: 1-D array containing the outer edges of radial bins
    :type rout_m: numpy.ndarray
    :param dens: Theano tensor including the density profile evaluated at the chosen radial bins
    :type dens: theano.tensor
    :return: Cumulative gas mass profile
    :rtype: theano.tensor
    '''

    # Integration volumes
    volint = 4. / 3. * np.pi * (rout_m ** 3 - rin_m ** 3)

    nvalm = len(rin_m)

    ones_mat = np.ones((nvalm, nvalm))

    cs_mat = np.tril(ones_mat)

    mgas_out = pm.math.dot(cs_mat, dens * volint) / 1e13

    return mgas_out


def list_params(Mhyd, nbetas=6, min_beta=0.6, max_beta=3, nrc=None, min_rc=None, max_rc=None, ):
    """
    Define a list of parameters to define the dictionary of basis functions

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param nrc: Number of core radii. If nrc=None (default), the number of core radiis will be defined adaptively as one per each set of 4 data points
    :type nrc: int
    :param nbetas: Number of beta values. Defaults to 6
    :type nbetas: int
    :param min_beta: Minimum value of beta. Defaults to 0.6
    :type min_beta: float
    :return: Array containing sets of values to set up the function dictionary
    :rtype: numpy.ndarray
    """

    rfit = np.array([])
    if Mhyd.sb_data is not None:
        for this_sb_data in Mhyd.sb_data:
            rfit = np.append(rfit, this_sb_data.rref_x_am)
    npfit = len(rfit)


    if nrc is None:
        nrc = np.max([int(npfit / nsh), 1])
    if min_rc is None:
        min_rc = rfit[2]
    if max_rc is None:
        max_rc = rfit[npfit - 1] / 2.
    if nbetas is None:
        nbetas = 6
    if min_beta is None:
        min_beta = 0.34
    if max_beta is None:
        max_beta = 3


    allrc = np.logspace(np.log10(min_rc), np.log10(max_rc), nrc)
    allbetas = np.linspace(min_beta, max_beta, nbetas)
    nrc = len(allrc)
    nbetas = len(allbetas)
    rc = allrc.repeat(nbetas)
    betas = np.tile(allbetas, nrc)
    ptot = np.empty((nrc * nbetas, 2))
    ptot[:, 0] = betas
    ptot[:, 1] = rc
    return ptot


# def rads_more(Mhyd, nmore=5, extend=False):
#     """
#
#     Return grid of (in, out) radii from X-ray, SZ data or both. Concatenates radii if necessary, then computes a grid of radii.
#     Returns the output arrays and the indices corresponding to the input X-ray and/or SZ radii.
#
#     :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing loaded X-ray and/or SZ loaded data.
#     :type Mhyd: class:`hydromass.mhyd.Mhyd`
#     :param nmore: Number of subgrid values to compute the fine binning. Each input bin will be split into nmore values. Defaults to 5.
#     :type nmore: int
#     :return:
#         - rin, rout: the inner and outer radii of the fine grid
#         - index_x, index_sz: lists of indices corresponding to the position of the input values in the grid
#         - sum_mat: matrix containing the number of values in each subgrid bin
#         - ntm: total number of grid points
#     """
#
#     rout_joint = np.array([])  # Mhyd.sb_data.rout_x_kpc
#     rref_joint = np.array([])  # Mhyd.sb_data.rref_x_kpc
#
#     if Mhyd.spec_data is not None:
#         rout_joint = np.append(rout_joint, Mhyd.spec_data.rout_x_kpc)
#         rref_joint = np.append(rref_joint, Mhyd.spec_data.rref_x_kpc)
#
#     if Mhyd.sz_data is not None:
#         rout_joint = np.append(rout_joint, Mhyd.sz_data.rout_sz_kpc)
#         rref_joint = np.append(rref_joint, Mhyd.sz_data.rref_sz_kpc)
#
#     rout_joint = np.sort(np.unique(rout_joint))
#     rref_joint = np.sort(np.unique(rref_joint))
#
#     rin_joint = np.roll(rout_joint, 1)
#
#     rin_joint[0] = 0.
#
#     # rin_joint[0] = np.exp(np.log(rout_joint)[0] - np.diff(np.log(rout_joint))[0])
#
#     njoint = len(rref_joint)
#
#     tot_joint = np.sort(np.append(rin_joint, rref_joint))
#
#     ntotjoint = len(tot_joint)
#
#     ntm = int((ntotjoint - 0.5) * nmore)
#
#     rout_more = np.empty(ntm)
#
#     for i in range(ntotjoint - 1):
#         rout_more[i * nmore:(i + 1) * nmore] = np.linspace(tot_joint[i], tot_joint[i + 1], nmore + 1)[1:]
#
#     rout_more[(ntotjoint - 1) * nmore:] = np.linspace(rref_joint[njoint - 1], rout_joint[njoint - 1], int(nmore / 2.) + 1)[1:]
#
#     # Move the outer boundary to the edge of the SB profile if it is farther out
#
#     if Mhyd.max_rad > np.max(rout_more):
#         nvm = len(rout_more)
#
#         dx_out = rout_more[nvm - 1] - rout_more[nvm - 2]
#
#         rout_2add = np.arange(np.max(rout_more), Mhyd.max_rad, dx_out)
#
#         rout_2add = np.append(rout_2add[1:], Mhyd.max_rad)
#
#         rout_more = np.append(rout_more, rout_2add)
#
#         ntm = len(rout_more)
#
#     # ADD SB points
#     rout_more = np.sort(np.unique(np.append(rout_more, Mhyd.sb_data.rref_x_kpc)))
#     #
#
#     rin_more = np.roll(rout_more, 1)
#
#     rin_more[0] = 0
#
#     # rin_more[0] = np.exp(np.log(rout_more)[0] - np.diff(np.log(rout_more))[0])
#
#     index_x, index_sz = None, None
#
#     index_sb = np.where(np.in1d(rout_more, Mhyd.sb_data.rref_x_kpc))
#
#     if Mhyd.spec_data is not None:
#         index_x = np.where(np.in1d(rout_more, Mhyd.spec_data.rref_x_kpc))
#
#     if Mhyd.sz_data is not None:
#         index_sz = np.where(np.in1d(rout_more, Mhyd.sz_data.rref_sz))
#
#     sum_mat_kt = None
#
#     if Mhyd.spec_data is not None:
#
#         ntot = len(rout_more)
#
#         nx = len(Mhyd.spec_data.rref_x_kpc)
#
#         if not extend:
#
#             sum_mat_kt = np.zeros((nx, ntot))
#
#         else:
#
#             if Mhyd.spec_data.rout_x[nx - 1] < np.max(rin_more):
#                 sum_mat_kt = np.zeros((nx + 1, ntot))
#
#             else:
#                 sum_mat_kt = np.zeros((nx, ntot))
#
#         for i in range(nx):
#             ix = np.where(np.logical_and(rin_more < Mhyd.spec_data.rout_x_kpc[i], rin_more >= Mhyd.spec_data.rin_x_kpc[i]))
#
#             nval = len(ix[0])
#
#             sum_mat_kt[i, :][ix] = 1. / nval
#
#         if extend:
#
#             if Mhyd.spec_data.rout_x_kpc[nx - 1] < np.max(rin_more):
#                 ix = np.where(rin_more >= Mhyd.spec_data.rout_x_kpc[nx - 1])
#
#                 nval = len(ix[0])
#
#                 sum_mat_kt[nx, :][ix] = 1. / nval
#
#     sum_mat_sb = None
#
#     if Mhyd.sb_data is not None:
#
#         ntot = len(rout_more)
#
#         nx = len(Mhyd.sb_data.rref_x_kpc)
#
#         if not extend:
#
#             sum_mat_sb = np.zeros((nx, ntot))
#
#         else:
#
#             if Mhyd.sb_data.rout_x[nx - 1] < np.max(rin_more):
#                 sum_mat_sb = np.zeros((nx + 1, ntot))
#
#             else:
#                 sum_mat_sb = np.zeros((nx, ntot))
#
#         for i in range(nx):
#             ix = np.where(np.logical_and(rin_more < Mhyd.sb_data.rout_x_kpc[i], rin_more >= Mhyd.sb_data.rin_x_kpc[i]))
#
#             nval = len(ix[0])
#
#             sum_mat_sb[i, :][ix] = 1. / nval
#
#         if extend:
#
#             if Mhyd.sb_data.rout_x_kpc[nx - 1] < np.max(rin_more):
#                 ix = np.where(rin_more >= Mhyd.sb_data.rout_x_kpc[nx - 1])
#
#                 nval = len(ix[0])
#
#                 sum_mat_sb[nx, :][ix] = 1. / nval
#
#     return rin_more, rout_more, index_x, index_sz, sum_mat_kt, ntm, index_sb, sum_mat_sb


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
        - rin_p, rout_p: the fine grid extended with 2 * nmore shells out to 10 Mpc for the Compton y projection (None without SZ data)
        - sum_mat_sz_list: list of (integration matrix, area factor) from :func:`sum_mat_sz` on the extended grid, one per SZ data set
    """

    rout_joint = np.array([])
    rref_joint = np.array([])


    if Mhyd.sb_data is not None:
        for this_sb_data in Mhyd.sb_data:
            rout_joint = np.append(rout_joint, this_sb_data.rout_x_kpc)
            rref_joint = np.append(rref_joint, this_sb_data.rref_x_kpc)

    if Mhyd.spec_data is not None:
        for this_spec_data in Mhyd.spec_data:
            rout_joint = np.append(rout_joint, this_spec_data.rout_x_kpc)
            rref_joint = np.append(rref_joint, this_spec_data.rref_x_kpc)

    if Mhyd.sz_data is not None:
        for this_sz_data in Mhyd.sz_data:
            rout_joint = np.append(rout_joint, this_sz_data.rout_sz_kpc)
            rref_joint = np.append(rref_joint, this_sz_data.rref_sz_kpc)

    if rout_joint.size == 0:
        print('No loaded SZ or X-ray data found in input hydromass.Mhyd object, nothing to do')
        return

    rout_joint = np.sort(np.unique(np.round(rout_joint, decimals=6)))
    rref_joint = np.sort(np.unique(np.round(rref_joint, decimals=6)))

    rin_joint = np.roll(rout_joint, 1)
    rin_joint[0] = 0.

    njoint = len(rref_joint)

    tot_joint = np.sort(np.unique(np.round(np.append(rin_joint, rref_joint), decimals=6)))
    # unique needed if there is overlap in in-out or rref-rin
    # round needed for decimal rounding
    ntotjoint = len(tot_joint)

    ntail = max(1, int(nmore / 2.)) # fix added for nmore=1 case. nmore>=2 is the same

    ntm = (ntotjoint - 1) * nmore + ntail

    rout_more = np.empty(ntm)

    for i in range(ntotjoint - 1):

        rout_more[i * nmore:(i + 1) * nmore] = np.linspace(tot_joint[i], tot_joint[i + 1], nmore + 1)[1:]

    rout_more[(ntotjoint - 1) * nmore:] = np.linspace(tot_joint[-1], rout_joint[-1], ntail + 1)[1:]

    # Move the outer boundary to the edge of the SB profile if it is farther out
    if Mhyd.max_rad_kpc > np.max(rout_more):
        nvm = len(rout_more)

        dx_out = rout_more[nvm - 1] - rout_more[nvm - 2]

        rout_2add = np.arange(np.max(rout_more), Mhyd.max_rad_kpc, dx_out)

        rout_2add = np.append(rout_2add[1:], Mhyd.max_rad_kpc)

        rout_more = np.append(rout_more, rout_2add)

        ntm = len(rout_more)


    rin_more = np.roll(rout_more, 1)
    rin_more[0] = 0

    index_sb_list, index_kt_list, index_sz_list = [], [], []

    if Mhyd.sb_data is not None:
        for this_sb_data in Mhyd.sb_data:
            this_index_x = np.array([
                np.argmin(np.abs(rout_more - r)) for r in this_sb_data.rref_x_kpc
            ])

            # sanity check: matched radii should actually be close to the requested ones
            assert np.allclose(rout_more[this_index_x], this_sb_data.rref_x_kpc, atol=1e-4), \
                f"X-ray SB radius matching failed: {rout_more[this_index_x]} vs {this_sb_data.rref_x_kpc}"

            index_sb_list.append(this_index_x)

    if Mhyd.spec_data is not None:

        for this_spec_data in Mhyd.spec_data:

            this_index_x = np.array([
                np.argmin(np.abs(rout_more - r)) for r in this_spec_data.rref_x_kpc
            ])

            # sanity check: matched radii should actually be close to the requested ones
            assert np.allclose(rout_more[this_index_x], this_spec_data.rref_x_kpc, atol=1e-4), \
                f"X-ray kT radius matching failed: {rout_more[this_index_x]} vs {this_spec_data.rref_x_kpc}"

            index_kt_list.append(this_index_x)


    if Mhyd.sz_data is not None:

        for this_sz_data in Mhyd.sz_data:

            this_index_sz = np.array([
                np.argmin(np.abs(rout_more - r)) for r in this_sz_data.rref_sz_kpc
            ])

            # sanity check: matched radii should actually be close to the requested ones
            assert np.allclose(rout_more[this_index_sz], this_sz_data.rref_sz_kpc, atol=1e-4), \
                f"SZ radius matching failed: {rout_more[this_index_sz]} vs {this_sz_data.rref_sz_kpc}"

            index_sz_list.append(this_index_sz)

    sum_mat_kt_list = []

    if Mhyd.spec_data is not None:

        ntot = len(rout_more)

        for this_spec_data in Mhyd.spec_data:

            nx = len(this_spec_data.rref_x_kpc)

            if not extend:

                sum_mat = np.zeros((nx, ntot))

            else:

                if this_spec_data.rout_x[nx - 1] < np.max(rin_more):
                    sum_mat = np.zeros((nx + 1, ntot))

                else:
                    sum_mat = np.zeros((nx, ntot))

            for i in range(nx):
                ix = np.where(np.logical_and(rin_more < this_spec_data.rout_x_kpc[i], rin_more >= this_spec_data.rin_x_kpc[i]))

                nval = len(ix[0])

                sum_mat[i, :][ix] = 1. / nval

            if extend:

                if this_spec_data.rout_x_kpc[nx - 1] < np.max(rin_more):
                    ix = np.where(rin_more >= this_spec_data.rout_x_kpc[nx - 1])

                    nval = len(ix[0])

                    sum_mat[nx, :][ix] = 1. / nval

            sum_mat_kt_list.append(sum_mat)

    sum_mat_sb_list = []

    for this_sb_data in Mhyd.sb_data:

        ntot = len(rout_more)

        nx = len(this_sb_data.rref_x_kpc)

        if not extend:

            sum_mat_sb = np.zeros((nx, ntot))

        else:

            if this_sb_data.rout_x[nx - 1] < np.max(rin_more):
                sum_mat_sb = np.zeros((nx + 1, ntot))

            else:
                sum_mat_sb = np.zeros((nx, ntot))

        for i in range(nx):
            ix = np.where(np.logical_and(rin_more < this_sb_data.rout_x_kpc[i], rin_more >= this_sb_data.rin_x_kpc[i]))

            nval = len(ix[0])

            sum_mat_sb[i, :][ix] = 1. / nval

        if extend:

            if this_sb_data.rout_x_kpc[nx - 1] < np.max(rin_more):
                ix = np.where(rin_more >= this_sb_data.rout_x_kpc[nx - 1])

                nval = len(ix[0])

                sum_mat_sb[nx, :][ix] = 1. / nval

        sum_mat_sb_list.append(sum_mat_sb)

    # Extended grid out to 10 Mpc for the projection of the pressure into the Compton y profile, and SZ bin integration matrices on it
    rin_more_p, rout_more_p, sum_mat_sz_list = None, None, []

    if Mhyd.sz_data is not None:

        nout = 2 * nmore

        rout_more_p = np.append(rout_more, np.logspace(np.log10(np.max(rout_more) * 1.1), np.log10(10000.), nout))
        rin_more_p = np.append(rin_more, rout_more_p[ntm - 1:ntm - 1 + nout])

        for this_sz_data in Mhyd.sz_data:

            sum_mat_sz_list.append(sum_mat_sz(this_sz_data, rin_more_p, rout_more_p))

    return rin_more, rout_more, index_kt_list, index_sz_list, sum_mat_kt_list, ntm, index_sb_list, sum_mat_sb_list, rin_more_p, rout_more_p, sum_mat_sz_list

def sum_mat_sz(sz_data, rin_m, rout_m):
    '''
    CLAUDE: Integration matrix for the SZ profile, adapted from :func:`hydromass.plots.sum_mat_sz` (upstream commit e2f59f4).
    Each row selects the fine-grid annuli whose inner edge falls in the corresponding SZ bin, weighted by their projected area,
    so that dot(sum_mat, y) / area_tot is the area-weighted average of y in each SZ bin.

    :param sz_data: SZ data set (one element of Mhyd.sz_data), with bin edges rin_sz_kpc and rout_sz_kpc
    :type sz_data: class:`hydromass.tpdata.SZData`
    :param rin_m: Inner radial bins in kpc
    :type rin_m: numpy.ndarray
    :param rout_m: Outer radial bins in kpc
    :type rout_m: numpy.ndarray
    :return: Integration matrix and area factor
    :rtype: numpy.ndarray
    '''
    ntot = len(rout_m)

    nsz = len(sz_data.rref_sz_kpc)

    sum_mat = np.zeros((nsz, ntot))

    area_proj = np.pi * (rout_m ** 2 - rin_m ** 2)

    area_tot = np.empty(nsz)

    for i in range(nsz):
        ix = np.where(np.logical_and(rin_m < sz_data.rout_sz_kpc[i], rin_m >= sz_data.rin_sz_kpc[i]))

        sum_mat[i, :][ix] = area_proj[ix]

        area_tot[i] = np.sum(area_proj[ix])

    return sum_mat, area_tot

def cumsum_mat(nval):
    """

    Function to create a matrix that flips a vector, makes a cumulative sum and adds a 0 as the first element
    Then dot(mat, vector) returns the Riemann integral as a cumulative sum

    :param nval: Ve
    ctor size
    :type nval: int
    :return: Cumulative sum operator
    :type nval: numpy.ndarray
    """
    onemat = np.ones((nval - 1, nval - 1))
    triu = np.triu(onemat)
    flipped_triu = np.flipud(triu)
    zeromat = np.zeros((nval, nval))
    zeromat[1:, 1:] = flipped_triu
    totmat = np.flipud(zeromat)
    return totmat


def common_utils(Mhyd, nmore=5, rmax_amin=None, rmin_amin=None):
    if rmax_amin is None:
        rmax_amin = np.max(Mhyd.max_rad_am)
    if rmin_amin is None:
        rmin_amin = 0

    dict_common_utils = dict()
    rin_m, rout_m, index_kt_list, index_sz_list, sum_mat_kt_list, ntm, index_sb_list, sum_mat_sb_list, rin_m_p, rout_m_p, sum_mat_sz_list = rads_more(Mhyd, nmore=nmore)

    rref_m = (rin_m + rout_m) / 2.
    volmat = np.array(MyDeprojVol(rin_m / Mhyd.amin2kpc, rout_m / Mhyd.amin2kpc).deproj_vol().T)

    dict_common_utils['rin_m'] = rin_m
    dict_common_utils['rout_m'] = rout_m
    dict_common_utils['rref_m'] = rref_m

    dict_common_utils['index_kt_list'] = index_kt_list
    dict_common_utils['index_sz_list'] = index_sz_list
    dict_common_utils['sum_mat_kt_list'] = sum_mat_kt_list
    dict_common_utils['ntm'] = ntm
    dict_common_utils['index_sb_list'] = index_sb_list
    dict_common_utils['sum_mat_sb_list'] = sum_mat_sb_list
    dict_common_utils['rin_m_p'] = rin_m_p
    dict_common_utils['rout_m_p'] = rout_m_p
    dict_common_utils['sum_mat_sz_list'] = sum_mat_sz_list

    dict_common_utils['rmax_amin'] = rmax_amin
    dict_common_utils['rmin_amin'] = rmin_amin
    dict_common_utils['volmat'] = volmat

    n_r_m = len(rref_m)
    int_mat = cumsum_mat(n_r_m)
    dict_common_utils['int_mat'] = int_mat

    return dict_common_utils


def calc_sb_operator(rad, sourcereg, pars, withbkg=True):
    """
    Function to calculate a linear operator transforming a parameter vector into a model surface brightness profile

    .. math::

        S_X(r) = \\sum_{i=1}^P \\alpha_i S_i(r)

    with :math:`\\alpha_i` the parameter values and :math:`S_i(r)` the brightness profiles of each basis functions, i.e. the indices of the output matrix

    :param rad: Array of input radii in arcmin
    :type rad: numpy.ndarray
    :param sourcereg: Selection array for the source region
    :type sourcereg: numpy.ndarray
    :param pars: List of beta model parameters obtained through list_params
    :type pars: numpy.ndarray
    :param withbkg: Set whether the background is fitted jointly (True) or subtracted (False). Defaults to True.
    :type withbkg: bool
    :return: Linear projection operator
    :rtype: numpy.ndarray
    """

    # Select values in the source region
    rfit = np.array(rad[sourcereg])
    beta = pars[:, 0]
    rc = pars[:, 1]

    base = 1. + np.power(rfit[:, np.newaxis] / rc[np.newaxis, :], 2)
    expon = -3. * beta[np.newaxis, :] + 0.5
    func_base = np.power(base, expon)

    # Recast into full matrix and add column for background
    if withbkg:
        nptot = len(rad)
        npt = len(rfit)
        npars = len(pars[:, 0])
        Ktot = np.zeros((nptot, npars + 1))
        Ktot[0:npt, 0:npars] = func_base
        Ktot[:, npars] = 0.0

    else:
        Ktot = func_base

    return Ktot


def sb_utils(Mhyd, dict_utils, fit_bkg=False, bkglim_amin=None, backval=None, sb_setup={}):
    rmin_amin = dict_utils['rmin_amin']
    rmax_amin = dict_utils['rmax_amin']
    volmat = dict_utils['volmat']
    rin_m = dict_utils['rin_m']
    rout_m = dict_utils['rout_m']
    sum_mat_sb_list = dict_utils['sum_mat_sb_list']

    # Set vector with list of parameters
    pars = list_params(**sb_setup, Mhyd=Mhyd)
    npt = len(pars)

    pardens = copy.deepcopy(pars)
    pardens[:, 1] = pars[:, 1] * Mhyd.amin2kpc  # just the same but with rc in kpc

    # Define maximum radius for source deprojection, assuming we have only background for r>bkglim
    if bkglim_amin is None:
        bkglim_amin = 0
        for this_sb_data in Mhyd.sb_data:
            tmp_bkglim_amin = float(np.max(this_sb_data.rout_x_am))
            if tmp_bkglim_amin > bkglim_amin:
                bkglim_amin = tmp_bkglim_amin

    valid_sb_list = []
    testbkg_list = []
    proj_mat_sb_list = []
    Kdens_list = []
    K_list = []
    for this_sb_data, sum_mat_sb in zip(Mhyd.sb_data, sum_mat_sb_list):
        valid_sb = np.where(np.logical_and(this_sb_data.rref_x_am >= rmin_amin, this_sb_data.rref_x_am < rmax_amin))
        valid_sb_list.append(valid_sb)

        backreg = np.where(this_sb_data.rin_x_am > bkglim_amin)
        if backval is None:
            backval = np.mean(this_sb_data.sb[backreg])

        if not np.isfinite(backval):
            backval = this_sb_data.sb[-1]

        # Set source region
        sourcereg = np.where(this_sb_data.rout_x_am <= bkglim_amin)


        # Compute linear combination kernel
        if fit_bkg:

            Ksb = calc_sb_operator(this_sb_data.rref_x_am, sourcereg, pars, withbkg=True)

            K = calc_linear_operator(this_sb_data.rref_x_am, sourcereg, pars, this_sb_data.area, this_sb_data.exposure, this_sb_data.psfmat)  # transformation to counts

        else:

            Ksb = calc_sb_operator(this_sb_data.rref_x_am, sourcereg, pars, withbkg=False)

            K = np.dot(this_sb_data.psfmat, Ksb)
        K_list.append(K)

        if np.isnan(backval) or backval <= 0 or backval is None:
            testbkg = -10. + np.log(this_sb_data.mean_cf)
        else:
            testbkg = np.log(backval) + np.log(this_sb_data.mean_cf)
        testbkg_list.append(testbkg)

        if fit_bkg:
            Kdens = calc_density_operator(this_sb_data.rref_x_kpc, pardens)
        else:
            Kdens = calc_density_operator(this_sb_data.rref_x_kpc, pardens, withbkg=False)
        Kdens_list.append(Kdens)

        rref_m = (rin_m + rout_m) / 2.

        proj_mat_sb = np.dot(np.dot(this_sb_data.psfmat, sum_mat_sb), volmat)

        proj_mat_sb_list.append(proj_mat_sb)


    testvals = np.array([])
    # Set up initial values
    for this_sb_data in Mhyd.sb_data:
        if np.isnan(this_sb_data.sb[0]) or this_sb_data.sb[0] <= 0:
            testval = -10. + np.log(this_sb_data.mean_cf)
        else:
            testval = np.log(this_sb_data.sb[0] / npt) + np.log(this_sb_data.mean_cf)

        testvals= np.append(testvals, testval)

    testval = np.mean(testvals)

    if fit_bkg:
        Kdens_m = calc_density_operator(rref_m, pardens, withbkg=True)
        Kdens_m_out = calc_density_operator(rout_m, pardens, withbkg=True)
        Kdens_grad = calc_grad_operator(rout_m, pardens, withbkg=True)
    else:
        Kdens_m = calc_density_operator(rref_m, pardens, withbkg=False)
        Kdens_m_out = calc_density_operator(rout_m, pardens, withbkg=False)
        Kdens_grad = calc_grad_operator(rout_m, pardens, withbkg=False)

    dict_utils['valid_sb_list'] = valid_sb_list
    dict_utils['testval'] = testval
    dict_utils['testbkg'] = testbkg_list
    dict_utils['proj_mat_sb_list'] = proj_mat_sb_list

    dict_utils['npt'] = npt

    dict_utils['Kdens_list'] = Kdens_list
    dict_utils['K_list'] = K_list


    dict_utils['Kdens_m'] = Kdens_m
    dict_utils['Kdens_m_out'] = Kdens_m_out
    dict_utils['Kdens_grad'] = Kdens_grad



def spec_utils(Mhyd, dict_utils):
    rmin_amin = dict_utils['rmin_amin']
    rmax_amin = dict_utils['rmax_amin']
    volmat = dict_utils['volmat']
    sum_mat_kt_list = dict_utils['sum_mat_kt_list']

    valid_spec_list = []
    proj_mat_kt_list = []

    for this_spec_data, sum_mat_kt in zip(Mhyd.spec_data, sum_mat_kt_list):

        valid_spec = np.where(np.logical_and(this_spec_data.rref_x_am >= rmin_amin, this_spec_data.rref_x_am <= rmax_amin))
        valid_spec_list.append(valid_spec)

        proj_mat_kt = np.dot(np.dot(this_spec_data.psfmat, sum_mat_kt), volmat)
        proj_mat_kt_list.append(proj_mat_kt)

    dict_utils['valid_spec_list'] = valid_spec_list
    dict_utils['proj_mat_kt_list'] = proj_mat_kt_list


def gnfw_p0(x, pars):
    '''
    Generalized NFW function to estimate the pressure at the outer boundary, P0

    .. math::

        P_{gNFW}(r) = \\frac{P_0} {(r/r_s)^\\gamma (1+(r/rs)^\\alpha)^{(\\beta-\\gamma)/\\alpha}}

    :param x: Radius
    :type x: numpy.ndarray
    :param pars: Array containing the five parameters (P0, rs, alpha, beta, and gamma) of the gNFW function
    :type pars: numpy.ndarray
    :return: MassModel pressure
    :rtype: numpy.ndarray
    '''
    P0 = pars[0]
    rs = pars[1]
    alpha = pars[2]
    beta = pars[3]
    gamma = pars[4]
    t1 = np.power(x / rs, gamma)
    t2 = np.power(1. + np.power(x / rs, alpha), (beta - gamma) / alpha)
    return P0 / t1 / t2


def estimate_P0(Mhyd, dens='sb', outfile=None):
    '''
    Provide an estimate of the pressure at the outer boundary by fitting a rough gNFW profile to the data. The value of :math:`P_0` is the integration constant that enters when we integrate the hydrostatic equilibrium equation to predict the pressure profile,

    .. math::

        P(r) = P_0 + \\int_0^r \\frac{GM(<r^{\\prime})} {r^{\\prime 2}} dr^{\\prime}

    A rough electron density profile is estimated by fitting a double beta model profile to the surface brightness, and temperature deprojection is neglected. The resulting pressure profile is fitted with a gNFW profile using the scipy.minimize function and the best-fit function is used to extrapolate the pressure to the outer boundary to provide a rough estimate of :math:`P_0`.

    :param Mhyd: A :class:`hydromass.mhyd.Mhyd` object containing the loaded data
    :type Mhyd: class:`hydromass.mhyd.Mhyd`
    :param dens: Set whether we will deproject the surface brightness profile (dens='sb') or the normalization of the spectrum (dens='norm'). Defaults to 'sb'
    :type dens: str
    :return: Estimated value of :math:`P_0`
    :rtype: float
    '''
    print('estimate_P0 currently uses only first sb_data and spec_data!!!')

    spec_data = Mhyd.spec_data[0]

    if dens == 'sb' or spec_data.norm is None:
        sbprof = copy.deepcopy(Mhyd.sb_data[0])

        modbeta = pyproffit.Model(pyproffit.DoubleBeta)

        sbprof.bins = sbprof.rref_x_am
        sbprof.ebins = sbprof.rref_x_am - sbprof.rin_x_am
        sbprof.profile = sbprof.sb
        sbprof.eprof = sbprof.esb

        fitbeta = pyproffit.Fitter(model=modbeta, profile=sbprof, bkg=-12., beta=0.7, rc1=0.5, rc2=2., norm=-2, ratio=2)

        fitbeta.minuit.fixed['bkg'] = True

        fitbeta.Migrad()

        rc1 = np.abs(fitbeta.params['rc1']) * Mhyd.amin2kpc  # kpc
        rc2 = np.abs(fitbeta.params['rc2']) * Mhyd.amin2kpc

        beta = fitbeta.params['beta']
        norm = fitbeta.params['norm']
        ratio = fitbeta.params['ratio']

        cfact1 = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc1
        cfact2 = gamma(3 * beta) / gamma(3 * beta - 0.5) / np.sqrt(np.pi) / rc2

        rfit = sbprof.bins * Mhyd.amin2kpc

        t1 = (1. + np.power(rfit / rc1, 2)) ** (-3. * beta) * cfact1
        t2 = (1. + np.power(rfit / rc2, 2)) ** (-3. * beta) * cfact2

        dens_prof = np.sqrt(10 ** norm * (t1 + ratio * t2) / sbprof.mean_cf * Mhyd.transf)

        ne_interp = np.interp(spec_data.rref_x_am, sbprof.bins, dens_prof)

        p_interp = ne_interp * spec_data.temp_x
        ep_interp = ne_interp * spec_data.errt_x

    else:
        sbprof = copy.copy(Mhyd.sbprof)

        dat = sbprof.data
        cra = sbprof.cra
        cdec = sbprof.cdec

        prof_n = pyproffit.Profile(dat, center_choice='custom_fk5', center_ra=cra, center_dec=cdec,
                                   binsize=3., maxrad=11.)

        prof_n.bins = spec_data.rref_x_am
        prof_n.ebins = (spec_data.rout_x_am - spec_data.rin_x_am) / 2.
        prof_n.profile = spec_data.norm
        prof_n.eprof = spec_data.norm_lo
        prof_n.nbin = len(spec_data.rref_x_am)

        depr = pyproffit.Deproject(z=Mhyd.redshift, cf=1., profile=prof_n)

        depr.OnionPeeling()

        p_interp = depr.dens * spec_data.temp_x
        ep_interp = depr.dens * spec_data.errt_x

    pars_press = np.array([3.28, 1200., 1.33, 4.72, 0.59])

    def chi2_gnfw(pars):
        pars_press[0] = 10 ** pars[0]
        pars_press[1] = pars[1]
        mm = gnfw_p0(spec_data.rref_x_kpc, pars_press)
        chi2 = np.sum((p_interp - mm) ** 2 / ep_interp ** 2)
        return chi2

    bnds = ((None, None), (0, None))

    res = minimize(chi2_gnfw, np.array([-4, 1200.]), method='Nelder-Mead', bounds=bnds)

    pars_press[0] = 10 ** res['x'][0]
    pars_press[1] = res['x'][1]

    p0 = gnfw_p0(Mhyd.max_rad_kpc, pars_press)

    return p0

def forw_utils(Mhyd, dict_utils):
    pass


def gp_utils(Mhyd, dict_utils):
    pass


def mass_utils(Mhyd, dict_utils, mstar=None, p0_prior=None, p0_type='sb', ):
    rout_m = dict_utils['rout_m']
    rref_m = dict_utils['rref_m']
    if mstar is not None:
        r_mstar = mstar[:, 0]
        cum_mstar = mstar[:, 1]
        mstar_m = np.interp(rref_m, r_mstar, cum_mstar) / Mhyd.mfact0 / 1e13
        mstar_m_out = np.interp(rout_m, r_mstar, cum_mstar) / Mhyd.mfact0 / 1e13

    if mstar is not None:
        dict_utils['mstar_m_rref'] = mstar_m
        dict_utils['mstar_m_rout'] = mstar_m_out

    # Integration constant as a fitting (nuisance) parameter
    if p0_prior is not None:
        P0_est = p0_prior[0]
        err_P0_est = p0_prior[1]
    else:
        P0_est = estimate_P0(Mhyd=Mhyd, dens=p0_type)
        Mhyd.logger.info(f'Estimated value of P0: {P0_est}')
        err_P0_est = np.log(10) * P0_est  # P0_est # 1 in ln

    dict_utils['P0_est'] = P0_est
    dict_utils['err_P0_est'] = err_P0_est


def ccf_utils(Mhyd, dict_utils):
    dict_ccf_utils = dict()
    if Mhyd.conversion_factor.conversion_factor_type != 'fit':
        dict_ccf_utils['ccf'] = Mhyd.conversion_factor.ccf

    return dict_ccf_utils
