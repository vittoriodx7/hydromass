import logging
import sys
from importlib import resources
import numpy as np

__all__ = ['create_logger', 'warning']

def cov_to_corr(cov):
    """Decompose a covariance matrix into (std devs, correlation matrix)."""
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    corr = np.clip(corr, -1.0, 1.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return std, corr


def corr_to_cov(std, corr):
    """Recombine std devs and a correlation matrix back into a covariance matrix."""
    return corr * np.outer(std, std)


def is_pos_def(mat):
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


# def nearest_pd_correlation(corr, eps=1e-8):
#     """Project a near-PSD correlation matrix onto the nearest valid PD one."""
#     corr = (corr + corr.T) / 2.0
#     eigvals, eigvecs = np.linalg.eigh(corr)
#     eigvals_clipped = np.clip(eigvals, eps, None)
#     corr_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
#     d = np.sqrt(np.diag(corr_pd))
#     corr_pd = corr_pd / np.outer(d, d)
#     np.fill_diagonal(corr_pd, 1.0)
#     return corr_pd

def nearest_pd_correlation(corr, eps=1e-8):
    corr = (corr + corr.T) / 2.0
    eigvals, eigvecs = sp_eigh(corr)
    eigvals_clipped = np.clip(eigvals, eps, None)
    corr_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    # re-normalize diagonal back to exactly 1
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)
    return corr_pd


def fix_covariance(cov_matrix, jitter=1e-8, verbose=True):
    """Fix a non-PD covariance matrix via eigenvalue clipping in correlation space."""
    if is_pos_def(cov_matrix):
        if verbose:
            print("Covariance matrix already PD, no fix needed.")
        return cov_matrix

    std, corr = cov_to_corr(cov_matrix)
    corr_fixed = nearest_pd_correlation(corr, eps=jitter)
    cov_fixed = corr_to_cov(std, corr_fixed)

    if verbose:
        print("Fixed non-PD covariance matrix (correlation-space eigenclip).")
        print("Max abs change:", np.max(np.abs(corr_fixed - corr)))

    return cov_fixed

def create_logger(name = None):
    '''
    Generic function to create and return a logger object.
    '''
    if name is None:
        name = __package__

    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Fallback: Define the "default" here because nothing else exists
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(asctime)s @%(name)-25s "%(message)s"', datefmt = '%d-%m-%y %H:%M'))
        logger.addHandler(sh)

    return logger


def warning(text = None):
    if text is None:
        text = 'WARNING'
    WARNING = '\033[4m\033[1m\033[91m'  # UNDERLINE BOLD RED
    ENDC = '\033[0m'  # NORMAL
    return f'{WARNING}{text}{ENDC}'


def get_data_file_path(data_file):
    """
    Returns the absolute path to the required data files dynamically.
    """
    # __package__ gives the name of the current package (e.g., 'hydromass')
    pkg = __package__.split('.')[0]

    try:
        # Using the modern 'files' API
        ref = resources.files(pkg) / 'data' / data_file

        if not ref.is_file():
            raise IOError(f"Data file {data_file} not found in {pkg}/data/")

        # as_file handles cases where the package is inside a zip/egg
        with resources.as_file(ref) as path:
            return str(path.absolute())

    except (ImportError, TypeError, KeyError) as e:
        raise IOError(f"Could not read or find data file {data_file} in package {pkg}. Error: {e}")
