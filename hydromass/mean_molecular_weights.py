import numpy as np
import logging
import sys
from .pyutils import create_logger, get_data_file_path

__all__ = ['mean_molecular_weights']


def mean_molecular_weights(abund = 'aspl', Zs = 0.3):
    '''
    Calculation of mean molecular weights given an abundance table and a metallicity. The function reads the abundance table from a file located in the hydromass databse and that is copied from the XSPEC abundance table implementation. The abundance table is then used to calculate the mean molecular weight, the number ratio of electrons to ions, and the mean molecular weight per electron.

    :param abund: Name of abundance table to be used. Available abundance tables are 'feld', 'angr', 'aneb', 'grsa', 'wilm', 'lodd', 'aspl', 'lpgp', 'lpgs', and 'felc'. Defaults to 'aspl'.
    :type abund: str
    :param Zs: Metallicity of the plasma with respect to Solar. Defaults to 0.3.
    :type Zs: float
    :return: Mean molecular weight, ratio of number densities of electrons to ions, mean molecular weight per electron
    :type float, float, float
    '''

    file_abund = get_data_file_path('abundances.dat')

    logger = create_logger('mean_molecular_weights')

    logger.info(f'Calculating mean molecular weights, and electrons to proton ratio for {abund=}, {Zs=}')

    with open(file_abund, 'r') as fin:
        lin = fin.readlines()

    atomic_weights = np.array([1.008, 4.003, 6.941, 9.012, 10.811, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.086, 30.974,
                               32.066, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38])

    atomic_numbers = np.arange(1, 31)

    nele = len(atomic_weights)

    Ns = np.empty(nele)

    for nl, line in enumerate(lin):

        if abund in line and nl < 12:

            tl = line.split()

            for i in range(nele):
                Ns[i] = float(tl[i + 1])

    abvec = np.ones(nele) * Zs  # Abundance corrected for ratio to Solar metallicity

    abvec[0] = 1  # Abundance of Hydrogen and Helium is set to 1

    abvec[1] = 1

    tot = np.sum(Ns * atomic_weights * abvec)

    Xj = Ns * atomic_weights / tot

    X = Xj[0]

    Y = Xj[1]

    Z = np.sum(Xj[2:])

    logger.info(f'X: {X:.4f}')
    logger.info(f'Y: {Y:.4f}')
    logger.info(f'Z: {Z:.4f}')

    mup = 1 / np.sum((1. + atomic_numbers) / atomic_weights * abvec * Xj)  # Eq. 10.20 of Carroll

    logger.info(f'Mean molecular weight: {mup:.4f}')

    nhc = np.sum(atomic_numbers * Ns * abvec) / Ns[0]  # electrons / Hydrogen nuclei

    logger.info(f'Number ratio of electrons to H: {nhc:.4f}')

    mu_e = 1 / np.sum(atomic_numbers / atomic_weights * abvec * Xj)
    logger.info(f'Mean molecular weight per electron: {mu_e:.4f}')

    return abund, mup, nhc, mu_e
