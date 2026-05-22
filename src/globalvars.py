"""Global constants and reference mode parameter loader.

Provides solar p-mode frequencies/fwhm from Larson & Schou (MDI) for
constructing synthetic power spectra and testing.
"""

import logging
import os

import numpy as np

from . import logger

LOGGER = logger.create_logger_stream(__name__, logging.NOTSET)

# Sun-optimised observational parameters
_T = 2678460.  # total observing time (seconds)
_dt = 60.      # cadence (seconds)
_Nt = int(_T / _dt)
_homega = 2. * np.pi / _T
_omega_plus = np.arange((_Nt + 1) / 2) * _homega
_nu_plus = _omega_plus / (2. * np.pi)


class globalVars:
    """Solar mode parameters and interpolation utilities.

    Loads reference p-mode frequencies and linewidths from the Larson & Schou
    (2015) MDI data set, with optional gap-filling using Stahn (2010) fits.
    """

    T = _T
    dt = _dt
    Nt = _Nt
    homega = _homega
    omegaPlus = _omega_plus
    nuPlus = _nu_plus

    def __init__(self, obs_dir=None):
        self.dt = _dt
        self._package_dir = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )
        self._obs_dir = obs_dir  # set via init or load_data() dirname
        LOGGER.info("globalVars initialised")

    # ------------------------------------------------------------------
    # Public data loaders
    # ------------------------------------------------------------------

    def load_data(self, dataset="larson", lmax=3, obs_dir=None):
        """Load reference solar p-mode parameters.

        Parameters
        ----------
        dataset : {"larson", "refined"}
            "larson" — raw data from *Larson_Schou_MDI_2015.dat*.
            "refined" — pre-processed .npy files under ``obs_dir/../ps-fits/``.
        lmax : int
            Maximum harmonic degree (only used for ``refined``).
        obs_dir : str or None
            Directory containing ``Larson_Schou_MDI_2015.dat``.
            Falls back to ``self._obs_dir`` if given earlier.

        Returns
        -------
        ells, enns, nu, fwhm, sig_fwhm : np.ndarray
        """
        if obs_dir is None:
            obs_dir = self._obs_dir
        if obs_dir is None:
            raise FileNotFoundError(
                "No obs_dir provided. Pass obs_dir=... to load_data() "
                "or set it in the constructor."
            )

        data = np.genfromtxt(f"{obs_dir}/Larson_Schou_MDI_2015.dat")
        ells = data[:, 0].astype(int)
        enns = data[:, 1].astype(int)
        nu = data[:, 2] * 1e-6          # Hz
        fwhm = data[:, 4] * 1e-6        # Hz
        sig_fwhm = data[:, 10]

        if dataset == "larson":
            return ells, enns, nu, fwhm, sig_fwhm

        if dataset == "refined":
            # refined mode parameters saved as .npy under obs_dir/../ps-fits/
            save_dir = os.path.normpath(f"{obs_dir}/../ps-fits")
            ELLS = np.load(
                f"{save_dir}/fitted-ell-list-{lmax}.npy"
            ).astype(int)
            ENNS = np.load(
                f"{save_dir}/fitted-enn-list-{lmax}.npy"
            ).astype(int)
            NUS = np.load(f"{save_dir}/fitted-nu-list-mod-{lmax}.npy")
            FWHMS = np.load(f"{save_dir}/fitted-fwhm-list-mod-{lmax}.npy")

            SIG_FWHMS = []
            for idx in range(len(ELLS)):
                _ell, _enn = ELLS[idx], ENNS[idx]
                try:
                    mode_idx = np.where(
                        (ells == _ell) & (enns == _enn)
                    )[0][0]
                    SIG_FWHMS.append(sig_fwhm[mode_idx])
                except IndexError:
                    pass
            return ELLS, ENNS, NUS, FWHMS, np.array(SIG_FWHMS)

        raise ValueError(f"Unknown dataset '{dataset}'")

    # ------------------------------------------------------------------
    # Stahn (2010) polynomial interpolation — kept for reproducibility
    # Used to gap-fill missing (n,l) modes when building synthetic spectra.
    # ------------------------------------------------------------------

    @staticmethod
    def get_nunl_stahn(enn, ell):
        """Interpolated mode frequency from Stahn (2010) polynomial fit.

        Parameters
        ----------
        enn : int   radial order
        ell : int   spherical harmonic degree (0, 1, 2)

        Returns
        -------
        nu_nl : float   frequency in muHz
        """
        cil = np.array(
            [
                [3033.65, 3098.04, 9.05000],
                [134.850, 135.100, -0.2400],
                [0.12000, 0.13250, 0.0],
            ]
        )
        enn0 = 21
        npoly = np.array([(enn - enn0) ** i for i in range(3)])
        if ell < 2:
            return float(npoly @ cil[:, ell])
        delta = cil[0, 2] + cil[1, 2] * (enn - enn0)
        npoly2 = np.array([(enn + 1 - enn0) ** i for i in range(3)])
        return float(npoly2 @ cil[:, 0] - delta)

    @staticmethod
    def get_gammanl_stahn(enn, ell):
        """Interpolated mode linewidth from Stahn (2010) polynomial fit.

        Parameters
        ----------
        enn : int   radial order
        ell : int   spherical harmonic degree (ignored)

        Returns
        -------
        gamma_nl : float   FWHM in muHz
        """
        gi = np.array([1.2, 0.177, 0.08, 0.0167])
        enn0 = 21
        npoly = np.array([(enn - enn0) ** i for i in range(len(gi))])
        return float(npoly @ gi)

    # ------------------------------------------------------------------
    # Gap-filled mode list (combines observed + Stahn-interpolated)
    # ------------------------------------------------------------------

    def gapfilled_nl(self, lmax=3, nmax=31, obs_dir=None):
        """Build a complete (ell, enn) mode list, filling gaps with Stahn
        polynomial frequencies / linewidths.

        Parameters
        ----------
        lmax : int    maximum harmonic degree
        nmax : int    maximum radial order to include
        obs_dir : str or None   passed to :meth:`load_data`

        Returns
        -------
        ell_list, enn_list, nu_list, fwhm_list : list
        """
        ells, enns, nu, fwhm, sig_fwhm = self.load_data(
            obs_dir=obs_dir
        )
        ell_list, enn_list, nu_list, fwhm_list = [], [], [], []

        for ell in range(lmax):
            mask = ells == ell
            enn_ell, nu_ell, fwhm_ell = enns[mask], nu[mask], fwhm[mask]
            for enn in range(int(enn_ell.min()), nmax):
                exists = (enn_ell == enn).sum()
                if exists:
                    idx = np.where(enn_ell == enn)[0][0]
                    ell_list.append(ell)
                    enn_list.append(enn_ell[idx])
                    nu_list.append(nu_ell[idx])
                    fwhm_list.append(fwhm_ell[idx])
                else:
                    ell_list.append(ell)
                    enn_list.append(enn)
                    nu_list.append(self.get_nunl_stahn(enn, ell) * 1e-6)
                    fwhm_list.append(self.get_gammanl_stahn(enn, ell) * 1e-6)

        return ell_list, enn_list, nu_list, fwhm_list