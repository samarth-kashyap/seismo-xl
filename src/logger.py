"""Misc functions needed for the module"""
import logging

def create_logger_file(logger_name, logger_file, logger_level):
    """Create a file-backed logger with a given name and level.

    Parameters
    ----------
    logger_name : str
        Name of the logger (typically ``__name__`` of the calling module).
    logger_file : str
        Path to the file where log messages are written.
    logger_level : int
        Logging level. One of ``logging.NOTSET``, ``logging.DEBUG``,
        ``logging.INFO``, ``logging.WARNING``, ``logging.ERROR``, or
        ``logging.CRITICAL``.

    Returns
    -------
    logger : logging.Logger
        Configured logger that writes to ``logger_file``.
    """
    logger = logging.getLogger(logger_name)
    filehandler = logging.FileHandler(logger_file)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    logger.setLevel(logger_level)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def create_logger_stream(logger_name, logger_level=None):
    """Create a stream (console) logger with a given name and level.

    Parameters
    ----------
    logger_name : str
        Name of the logger (typically ``__name__`` of the calling module).
    logger_level : int or None, optional
        Logging level. One of ``logging.NOTSET``, ``logging.DEBUG``,
        ``logging.INFO``, ``logging.WARNING``, ``logging.ERROR``, or
        ``logging.CRITICAL``.

    Returns
    -------
    logger : logging.Logger
        Configured logger that writes to ``sys.stderr``.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    if not logger.hasHandlers():
        sh = logging.StreamHandler()
        sh.setLevel(logger_level)
        formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger
