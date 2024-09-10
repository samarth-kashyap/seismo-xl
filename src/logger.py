"""Misc functions needed for the module"""
import logging

def create_logger_file(logger_name, logger_file, logger_level):
    """Creates a logger with a given name and specified logger level.

    Parameters
    ----------
    logger_name : str
        name of the logger
    logger_file : str
        file name of the logger
    logger_level :
        takes one of
        (logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL)

    Returns
    -------
    logger
    """
    logger = logging.getLogger(logger_name)
    filehandler = logging.FileHandler(logger_file)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    logger.setLevel(logger_level)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def create_logger_stream(logger_name, logger_level=None):
    """Creates a logger with a given name and specified logger level.

    Parameters
    ----------
    logger_name : str
        name of the logger
    logger_file : str
        file name of the logger
    logger_level :
        takes one of
        (logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL)

    Returns
    -------
    logger
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
