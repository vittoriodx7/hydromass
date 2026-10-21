import logging
import sys
from importlib import resources

__all__ = ['create_logger', 'warning']


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
