__all__ = ["stellarPS",
           "visibilityMatrix"]

__authors__ = ["samarth-kashyap"]

# Loading python modules
import logging
import numpy as np
from scipy.special import factorial
from pyshtools import legendre
from ritzLavelyPy.rlclass import ritzLavelyPoly

# Loading custom modules
from . import logger

# Creating logger
LOGGER = logger.create_logger_stream(__name__, logging.NOTSET)


class stellarPS():
    """Class for constructing solar power spectrum."""
    __methods__ = ["lorentzian",
                   "construct_ps_list",
                   "construct_ps_normed_nlm"]

    def __init__(self,
                 freqarr=np.arange(10)*0.01,
                 lmax=3,
                 incl_angle=0.,
                 mode_ell=None,
                 mode_enn=None,
                 mode_nu=None,
                 mode_fwhm=None,
                 mode_sigfwhm=None,
                 obs_spectrum=None,
                 a1rot=435.e-9,
                 a3rot=-10.7e-9,
                 fwhm_scale=1.0,
                 mode_min_nu=0.6e-3,
                 mode_max_nu=4.0e-3,):
        """Initialise a stellarPS instance.

        Parameters
        ----------
        freqarr : np.ndarray, optional
            Frequency array in Hz over which the power spectrum is evaluated.
        lmax : int, optional
            Maximum spherical harmonic degree.
        incl_angle : float, optional
            Stellar inclination angle in radians.
        mode_ell : np.ndarray or None, optional
            Array of spherical harmonic degrees for each mode.
        mode_enn : np.ndarray or None, optional
            Array of radial orders for each mode.
        mode_nu : np.ndarray or None, optional
            Array of mode frequencies in Hz.
        mode_fwhm : np.ndarray or None, optional
            Array of mode line-widths (FWHM) in Hz.
        mode_sigfwhm : np.ndarray or None, optional
            Array of uncertainties on mode line-widths in Hz.
        obs_spectrum : np.ndarray or None, optional
            Observed power spectrum (used for comparison).
        a1rot : float, optional
            First-order rotational splitting coefficient in Hz.
        a3rot : float, optional
            Third-order rotational splitting coefficient in Hz.
        fwhm_scale : float, optional
            Multiplicative scale factor applied to ``mode_fwhm``.
        mode_min_nu : float, optional
            Lower frequency cut-off for modes in Hz.
        mode_max_nu : float, optional
            Upper frequency cut-off for modes in Hz.
        """
        super(stellarPS, self).__init__()
        self.nu_plus  = freqarr*1.
        self.omega_plus = self.nu_plus*2*np.pi
        self.lmax = lmax
        self.incl_angle = incl_angle
        self.cosi = np.cos(incl_angle)
        self.a1rot = a1rot
        self.a3rot = a3rot
        self.acoeffs = np.array([0., a1rot, 0., a3rot])
        self.rot_rate = self.acoeffs[1]
        self.mode_ell = mode_ell
        self.mode_enn = mode_enn
        self.mode_nu = mode_nu
        if mode_fwhm is None:
            self.mode_fwhm = None
        else:
            self.mode_fwhm = mode_fwhm*fwhm_scale
        self.mode_sigfwhm = mode_sigfwhm
        self.obs_spectrum = obs_spectrum
        self.mode_min_nu = mode_min_nu
        self.mode_max_nu = mode_max_nu

    def lorentzian(self, nuc, gamma,):
        """Returns the Lorentzian profile for a given frequency,
        and damping, for a given range of frequencies.

        Parameters
        ----------
        nuc : np.float64
            location of peak (in Hz)
        gamma: np.float64
            damping rate (in Hz)

        Returns
        --------
            lorentzian profile - in Hz^{-2}
        """
        dxg1 = (self.nu_plus - nuc)/(gamma/2)
        lor1 = 1./(dxg1**2 + 1.0)
        return lor1


    def construct_ps_list(self, ell=1, visibility_matrix=True,
                          return_nl_list=False, shiftfreq=0.0, shiftenn=-10):
        """Construct a list of mode power spectra for a given harmonic degree.

        Parameters
        ----------
        ell : int, optional
            Spherical harmonic degree of the modes to include.
        visibility_matrix : bool, optional
            If ``True``, weight each (n, ell, m) Lorentzian by the geometrical
            visibility factor from Gizon & Solanki (2003).
        return_nl_list : bool, optional
            If ``True``, also return per-mode metadata lists.
        shiftfreq : float, optional
            Frequency offset applied to a single radial order ``shiftenn`` in Hz.
        shiftenn : int, optional
            Radial order to which ``shiftfreq`` is applied.

        Returns
        -------
        ps_list : list of np.ndarray
            m-summed Lorentzian power spectrum for each (n, ell) mode.
        enn_list : list of int
            Radial orders (only when ``return_nl_list`` is ``True``).
        ell_list : list of int
            Harmonic degrees (only when ``return_nl_list`` is ``True``).
        nu_list : list of float
            Mode frequencies in Hz (only when ``return_nl_list`` is ``True``).
        gamma_list : list of float
            Mode line-widths in Hz (only when ``return_nl_list`` is ``True``).
        """
        mask_ell = self.mode_ell==ell
        enn_ell = self.mode_enn[mask_ell]
        nu_ell = self.mode_nu[mask_ell]
        fwhm_ell = self.mode_fwhm[mask_ell]
        num_modes = len(nu_ell)
        ps_list = []
        ps_nlm_list = []
        enn_list = []
        ell_list = []
        nu_list = []
        gamma_list = []

        if ell>1:
            RLP = ritzLavelyPoly(ell, jmax=3)
            delnu_nlm = RLP.polyval(self.acoeffs)
        else:
            delnu_nlm = np.arange(-ell, ell+1)*self.a1rot

        if visibility_matrix:
            vsm = visibilityMatrix(incl_angle=self.incl_angle)
            LOGGER.info("Using visibility matrix")
            for i in range(num_modes):
                psl = 0.0
                _pslm = []
                for idxm, emm in enumerate(range(-ell, ell+1)):
                    nu_lm = nu_ell[i] + delnu_nlm[idxm]
                    if shiftenn==enn_ell[i]: nu_lm += shiftfreq
                    _elm = vsm.get_elm(ell, emm)
                    _ps = _elm * self.lorentzian(nu_lm, fwhm_ell[i])
                    _pslm.append(_ps)
                    psl = psl + _ps
                ps_nlm_list.append(_pslm)#/psl.max())
                ps_list.append(psl)#/psl.max())
                ell_list.append(ell)
                enn_list.append(enn_ell[i])
                nu_list.append(nu_ell[i])
                gamma_list.append(fwhm_ell[i])
        else:
            LOGGER.info("NOT Using visibility matrix")
            for i in range(num_modes):
                psl = self.lorentzian(nu_ell[i], fwhm_ell[i])
                ps_list.append(psl)
                ell_list.append(ell)
                enn_list.append(enn_ell[i])
        if return_nl_list:
            return (ps_list, ps_nlm_list), enn_list, ell_list, nu_list, gamma_list
        else:
            return ps_list

        
    def construct_ps_normed_nlm(self, enn=10, ell=1, shiftfreq=0.0, scalefwhm=1.0,):
        """Construct per-m Lorentzians weighted by the visibility matrix for a single mode.

        Parameters
        ----------
        enn : int, optional
            Radial order of the mode.
        ell : int, optional
            Spherical harmonic degree of the mode.
        shiftfreq : float, optional
            Frequency offset added to the mode centre frequency in Hz.
        scalefwhm : float, optional
            Multiplicative scale factor applied to the mode line-width.

        Returns
        -------
        ps_nlm : list of np.ndarray
            List of length ``2*ell+1`` containing the visibility-weighted
            Lorentzian for each azimuthal order m in [-ell, ell].
        """
        
        mask_ell = self.mode_ell==ell
        mask_enn = self.mode_enn==enn
        mask_all = mask_ell * mask_enn
        nu0 = self.mode_nu[mask_all] + shiftfreq
        fwhm0 = self.mode_fwhm[mask_all]*scalefwhm

        ps_nlm = []

        if ell>1:
            RLP = ritzLavelyPoly(ell, jmax=3)
            delnu_nlm = RLP.polyval(self.acoeffs)
        else:
            delnu_nlm = np.arange(-ell, ell+1)*self.a1rot

        vsm = visibilityMatrix(incl_angle=self.incl_angle)
        LOGGER.info("Using visibility matrix")
        for idxm, emm in enumerate(range(-ell, ell+1)):
            nu_m = nu0 + delnu_nlm[idxm]
            _elm = vsm.get_elm(ell, emm)
            _ps = _elm * self.lorentzian(nu_m, fwhm0)
            ps_nlm.append(_ps)
        return ps_nlm



class visibilityMatrix(stellarPS):
    """Class to generate visibility-matrix for computing change in mode-visibility
    due to inclination angle."""
    __attributes__ = ["cosi"]
    __methods__ = ["get_elm",
                   "get_elm_gs"]

    def __init__(self, incl_angle=0., degrees=False):
        """Initialise a visibilityMatrix instance.

        Parameters
        ----------
        incl_angle : float, optional
            Stellar inclination angle. Interpreted as radians unless
            ``degrees=True``.
        degrees : bool, optional
            If ``True``, ``incl_angle`` is given in degrees and is converted
            to radians internally.
        """
        super(visibilityMatrix, self).__init__()
        if degrees:
            incl_angle = incl_angle*np.pi/180.
        elif abs(incl_angle) > np.pi/2:
            LOGGER.warning("Inclination angle should be provided in radians")
        self.cosi = np.cos(incl_angle)

    def get_elm(self, ell, emm):
        """Gets the visibility matrix for the given inclination angle.

        Inputs
        ------
        ell : int
            Spherical harmonic degree of mode
        emm : int
            abs(emm) <= ell
            azimuthal order of the mode

        Returns
        -------
        elm : float
            visibility matrix, as defined by Eqn.(11) of Gizon and Solanki (2003)
        """
        Plmfunc = legendre.PLegendreA
        _Plm = Plmfunc(ell, self.cosi)
        _idx = legendre.PlmIndex(ell, abs(emm))
        pval = _Plm[_idx]
        lpm = ell + abs(emm)
        lmm = ell - abs(emm)
        lpmf = factorial(lpm)
        lmmf = factorial(lmm)
        pval = pval * np.sqrt(lmmf/lpmf)
        elm = pval**2
        return elm

    def get_elm_gs(self, l, m):
        """Exact form of visibility matrix as given by Gizon and Solanki (2003) -- Eqns(12-16)

        Inputs
        ------
        l : int
            Spherical harmonic degree of mode
        m : int
            abs(m) <= l
            azimuthal order of the mode

        Returns
        -------
            visibility matrix, as defined by Eqn.(12-16) of Gizon and Solanki (2003)
        """
        x = self.cosi
        if   (l==0) and (abs(m)==0): return 1.
        elif (l==1) and (abs(m)==0): return x**2
        elif (l==1) and (abs(m)==1): return 0.5*(1-x**2)
        elif (l==2) and (abs(m)==0): return 0.25*(3*x**2 -1.)**2
        elif (l==2) and (abs(m)==1): return 1.5*(x**2)*(1-x**2)
        elif (l==2) and (abs(m)==2): return 3./8.*(1.-x**2)**2
