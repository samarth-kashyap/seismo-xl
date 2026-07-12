"""config utilities for yml file."""
import os
import yaml
import logging

logger = logging.getLogger(__name__)


class LoaderMeta(type):
    """Constructor for supporting `!include`."""

    def __new__(mcs, __name__, __bases__, __dict__):
        """Add include constructer to class."""
        # register the include constructor on the class
        cls = super().__new__(mcs, __name__, __bases__, __dict__)
        cls.add_constructor('!include', cls.construct_include)
        return cls


class Loader(yaml.Loader, metaclass=LoaderMeta):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip('.')
        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                return yaml.load(f, Loader)
            else:
                return ''.join(f.readlines())


class DictAsMember(dict):
    """Dict subclass that exposes keys as attribute access."""

    def __getattr__(self, name):
        """Return the value for ``name``, wrapping nested dicts in DictAsMember."""
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


class Config(dict):
    """Config with yaml file.
    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.
    Args:
        filename(str): File name.
    Examples:
        yaml file ``config.yml``::
            Navg: 100
            Nmcmc: 1000
            data_dir: '/mnt/data/virgo'
        Usage in .py:
        >>> from src.config import Config
        >>> config = Config('config.yml')
        >>> print(config.Navg)
            100
        >>> print(config.Nmcmc)
            1000
        >>> print(config.data_dir)
            /mnt/data/virgo
    """

    def __init__(self, filename=None):
        """Load configuration from a YAML file.

        Parameters
        ----------
        filename : str
            Path to the YAML configuration file. The file must exist.
        """
        assert os.path.exists(filename), "ERROR: Config File doesn't exist."
        try:
            with open(filename, 'r') as f:
                self._cfg_dict = yaml.load(f, Loader)
        # parent of IOError, OSError *and* WindowsError where available
        except EnvironmentError:
            logger.error('Please check the file with name of "%s"', filename)
        print(' APP CONFIG '.center(80, '-'))
        self.show()
        print(''.center(80, '-'))

    def __getattr__(self, name):
        """Return the config value for ``name``, wrapping nested dicts in DictAsMember."""
        value = self._cfg_dict[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

    def show(self, cfg_dict=None, indent=0):
        """Recursively print configuration key-value pairs to stdout.

        Parameters
        ----------
        cfg_dict : dict or None, optional
            Sub-dictionary to display. Defaults to the top-level config.
        indent : int, optional
            Current indentation level (used for recursive calls).

        Returns
        -------
        int
            Updated indentation level after printing the current level.
        """
        if cfg_dict is None:
            cfg_dict = self._cfg_dict
        for key in cfg_dict:
            value = cfg_dict[key]
            if isinstance(value, dict):
                str_list = ['  '] * indent
                str_list.append(str(key))
                str_list.append(': ')
                print(''.join(str_list))
                indent = indent + 1
                indent = self.show(value, indent)
            else:
                str_list = ['  '] * indent
                str_list.append(str(key))
                str_list.append(': ')
                str_list.append(str(value))
                print(''.join(str_list))
        return indent - 1