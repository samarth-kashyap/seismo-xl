__all__ = ["stellarPS",
           "refWavelets",
           "csCorrection",
           "greensFunctions",
           "visibilityMatrix",
           "GVARS"]

__authors__ = ["samarth-kashyap"]

# Loading python modules
import sys
import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf
from pyshtools import legendre
from ritzLavelyPy.rlclass import ritzLavelyPoly

# Loading custom modules
from . import globalvars
from . import logger

GVARS = globalvars.globalVars()

# Creating logger
LOGGER = logger.create_logger_stream(__name__, logging.NOTSET)


def get_normed_cc(tmax_plot, tmin, tmax, C):
    return C[:tmax_plot]/np.amax(C[tmin:tmax])


class stellarPS():
    """Class for constructing solar power spectrum."""
    __attributes__ = ["lmax",
                      "incl_angle",
                      "cosi",
                      "mode_ell",
                      "mode_nu",
                      "mode_fwhm",
                      "mode_sigfwhm"]
    __methods__ = ["lorentzian",
                   "ps_from_lor",
                   "construct_ps_full",
                   "ps_envelope",
                   "get_background_lowfreq",
                   "get_background_midfreq"]

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
                 include_mgfcorr=False,
                 mgf_corr_freq=None,
                 mgf_corr_nl=None,
                 a1rot=435.e-9,
                 a3rot=-10.7e-9,
                 fwhm_scale=1.0,
                 mode_min_nu=0.6e-3,
                 mode_max_nu=4.0e-3,):
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
        self.include_mgfcorr = include_mgfcorr
        self.mode_min_nu = mode_min_nu
        self.mode_max_nu = mode_max_nu

        if self.include_mgfcorr:
            # Reading the array of frequency corrections due to
            # magnetic field perturbations
            self.mgf_corr_freq = mgf_corr_freq #nHz
            self.mgf_corr_nl = mgf_corr_nl


    def sigmoid(self, x, sigw=0.15):
        """Returns a sigmoid function over a given range.

        Parameters
        ----------
        :x: range over which sigmoid is computed
        :type: np.ndarray(ndim=1, dtype=float)

        :sigw: width of sigmoid
        :type: float

        Returns
        -------
        sigmoid function
        """
        return 1./(1. + np.exp(-x/sigw))

    def lorentzian(self, nuc, gamma, asymfac=1., sigw=0.15):
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
        # dxg2 = asymfac*(self.nu_plus - nuc)/(gamma/2)
        # lor2 = 1./(dxg2**2 + 1.0)
        # sigval = self.sigmoid(dxg1, sigw=sigw)
        # return lor2*sigval + lor1*(1 - sigval)

    def ps_from_lor(self, nuc, gamma):
        """Get power-spectrum from lorentzian profile.

        Parameters
        ----------
        nuc : np.float64
            location of peak (in Hz)
        gamma : np.float64
            damping rate (in Hz)

        Returns
        --------
            Power spectrum
        """
        return self.lorentzian(nuc, gamma, asymfac=1.0)

    def get_mgfcorr(self, enn, ell):
        """Get the correction to frequency from magnetic field"""
        if self.include_mgfcorr:
            try:
                idx = np.where((self.mgf_corr_nl[:, 0]==enn)*
                               (self.mgf_corr_nl[:, 1]==ell))[0][0]
                mgfcorr = self.mgf_corr_freq[idx] * 1e-9 # converting from nHz to Hz
            except IndexError:
                return np.zeros(2*ell+1)
        else:
            # mgfcorr = None
            # If no correction is present, 0.0 should be returned
            mgfcorr = 0.0
        return mgfcorr

    def construct_ps_ell(self, ell=1, visibility_matrix=True):
        """Constructs power spectrum using sum of Lorentzians for a specified ell.
        The mode-frequencies are read-off from observation table.
        The m-dependent frequency splitting due to rotation is modelled as m\Omega.
        The visibility matrix is used to scale lorentzians corresponding to different m.

        Parameters
        ----------
        ell : int
            Spherical harmonic degree

        Returns
        -------
        psl : np.ndarray(ndim=1)
            Array containing the power-spectrum constructed using specified ell.
        """
        psl = 0.0
        if visibility_matrix:
            vsm = visibilityMatrix(incl_angle=self.incl_angle)
        mask_ell = self.mode_ell==ell
        nu_ell = self.mode_nu[mask_ell]
        enn_ell = self.mode_enn[mask_ell]
        fwhm_ell = self.mode_fwhm[mask_ell]
        num_modes = len(nu_ell)

        if ell>1:
            RLP = ritzLavelyPoly(ell, jmax=3)
            delnu_nlm = RLP.polyval(self.acoeffs)
        else:
            delnu_nlm = np.arange(-ell, ell+1)*self.a1rot

        if visibility_matrix:
            LOGGER.info("Using visibility matrix")
            for idxm, emm in enumerate(range(-ell, ell+1)):
                _elm = vsm.get_elm(ell, emm)
                for i in range(num_modes):
                    nu_lm = nu_ell[i] + delnu_nlm[idxm]
                    psl += _elm * self.ps_from_lor(nu_lm, fwhm_ell[i])
        else:
            LOGGER.info("NOT Using visibility matrix")
            for i in range(num_modes):
                psl += self.ps_from_lor(nu_ell[i], fwhm_ell[i])

        return psl

    def get_nunl_stahn(self, enn, ell):
        """Returns the frequencies from the global power-spectral fitting performed by
        Stahn (2010).

        Parameters
        ----------
        :enn: radial order
        :type: int

        :ell: spherical harmonic degree
        :type: int

        Returns
        -------
        :nu_nl: mode-frequency in muHz
        :type: float
        """
        cil = np.zeros((3, 3), dtype=np.float64)
        cil[0, 0] = 3033.65 #muHz
        cil[0, 1] = 3098.04 #muHz
        cil[0, 2] = 9.05000 #muHz
        cil[1, 0] = 134.850 #muHz
        cil[1, 1] = 135.100 #muHz
        cil[1, 2] = -0.2400 #muHz
        cil[2, 0] = 0.12000 #muHz
        cil[2, 1] = 0.13250 #muHz

        enn0 = 21  # value used for fitting by Stahn
        if ell<2:
            npoly = [(enn - enn0)**i for i in range(3)]
            nu_nl = npoly @ cil[:, ell]
        else:
            delta_nu02 = cil[0, 2] + cil[1, 2]*(enn - enn0)
            npoly = [(enn + 1 - enn0)**i for i in range(3)]
            nu_np0 = npoly @ cil[:, 0]
            nu_nl = nu_np0 - delta_nu02
        return nu_nl

    def get_gammanl_stahn(self, enn, ell):
        """Returns the frequencies from the global power-spectral fitting performed by
        Stahn (2010).

        Parameters
        ----------
        :enn: radial order
        :type: int

        :ell: spherical harmonic degree
        :type: int

        Returns
        -------
        :nu_nl: mode-frequency in muHz
        :type: float
        """
        gi = np.array([1.2, 0.177, 0.08, 0.0167]) #muHz
        enn0 = 21
        # ?? This is done to prevent blowing up of linewidths
        if enn>25: gi[2:] = 0.
        npoly = [(enn - enn0)**i for i in range(len(gi))]
        gamma_nl = npoly @ gi
        return gamma_nl


    def get_gapfilled_nunl(self, ell, nmax=31):
        mask_ell = self.mode_ell==ell
        enn_ell = self.mode_enn[mask_ell].astype('int')
        nu_ell = self.mode_nu[mask_ell]
        fwhm_ell = self.mode_fwhm[mask_ell]

        if ell>2:
            return enn_ell, nu_ell, fwhm_ell
        else:
            nu_list = []
            enn_list = []
            fwhm_list = []
            for enn in range(enn_ell.min(), nmax, 1):
                nexists = (enn_ell==enn).sum()
                if nexists > 0:
                    idx = np.where(enn_ell==enn)[0][0]
                    nu_list.append(nu_ell[idx])
                    enn_list.append(enn_ell[idx])
                    fwhm_list.append(fwhm_ell[idx])
                else:
                    enn_list.append(enn)
                    nu_nl = self.get_nunl_stahn(enn, ell)*1e-6       #Hz
                    gamma_nl = self.get_gammanl_stahn(enn, ell)*1e-6 #Hz
                    nu_list.append(nu_nl)
                    fwhm_list.append(gamma_nl)
            return np.array(enn_list), np.array(nu_list), np.array(fwhm_list)


    def construct_ps_list(self, ell=1, visibility_matrix=True, stahn=True, return_nl_list=False, shiftfreq=0.0, shiftenn=-10):
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
                    if self.include_mgfcorr:
                        nu_mgf = self.get_mgfcorr(enn_ell[i], ell)[emm+ell]
                        nu_lm += nu_mgf
                    _ps = _elm * self.ps_from_lor(nu_lm, fwhm_ell[i])
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
                psl = self.ps_from_lor(nu_ell[i], fwhm_ell[i])
                ps_list.append(psl)#/psl.max())
                ell_list.append(ell)
                enn_list.append(enn_ell[i])
        if return_nl_list:
            return (ps_list, ps_nlm_list), enn_list, ell_list, nu_list, gamma_list
        else:
            return ps_list

        
    def construct_ps_normed_nlm(self, enn=10, ell=1, shiftfreq=0.0, scalefwhm=1.0, stahn=True):
        """To construct m-summed normalized lorentzians for the given visibility matrix.

        Parameters
        ----------
        enn, ell, shiftfreq, scalefwhm, stahn
        :enn: radial order of the mode
        :type: int
        :default: 10

        :ell: spherical harmonic degree of the mode
        :type: int
        :default: 1

        :shiftfreq: shifting mode frequency for fitting
        :type: float
        :default: 0.0
        :unit: microHz

        :scalefwhm: scaling FWHM for fitting
        :type: float
        :default: 1.0

        :stahn: flag to use Stahn's polynomials for mode frequency determination
        :type: bool
        :default: True

        Returns
        -------
        ps_nl

        :ps_nl: power spectrum constructed usign sum of lorentzians
        :type: np.ndarray(ndim=1, dtype=np.float)
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
            _ps = _elm * self.ps_from_lor(nu_m, fwhm0)
            ps_nlm.append(_ps)
        return ps_nlm




    def construct_ps_normed(self, enn=10, ell=1, shiftfreq=0.0, scalefwhm=1.0, stahn=True):
        """To construct m-summed normalized lorentzians for the given visibility matrix.

        Parameters
        ----------
        enn, ell, shiftfreq, scalefwhm, stahn
        :enn: radial order of the mode
        :type: int
        :default: 10

        :ell: spherical harmonic degree of the mode
        :type: int
        :default: 1

        :shiftfreq: shifting mode frequency for fitting
        :type: float
        :default: 0.0
        :unit: microHz

        :scalefwhm: scaling FWHM for fitting
        :type: float
        :default: 1.0

        :stahn: flag to use Stahn's polynomials for mode frequency determination
        :type: bool
        :default: True

        Returns
        -------
        ps_nl

        :ps_nl: power spectrum constructed usign sum of lorentzians
        :type: np.ndarray(ndim=1, dtype=np.float)
        """
        if not stahn or ell>2:
            mask_ell = self.mode_ell==ell
            mask_enn = self.mode_enn==enn
            mask_all = mask_ell * mask_enn
            if ell>3:
                _mask_nu = (self.mode_nu<self.mode_max_nu)*(self.mode_nu>self.mode_min_nu)
                mask_ell = mask_ell * _mask_nu
            nu0 = self.mode_nu[mask_all] + shiftfreq
            fwhm0 = self.mode_fwhm[mask_all]*scalefwhm
        else:
            enn_ell, nu_ell, fwhm_ell = self.get_gapfilled_nunl(ell)
            idx = np.argmin(abs(enn_ell - enn))
            nu0 = nu_ell[idx] + shiftfreq
            fwhm0 = fwhm_ell[idx]*scalefwhm

        ps_nl = 0.0

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
            _ps = _elm * self.ps_from_lor(nu_m, fwhm0)
            ps_nl = ps_nl + _ps
        return ps_nl

    def construct_ps_list_nlm(self, ell=1, visibility_matrix=True, stahn=True):
        if not stahn or ell>2:
            mask_ell = self.mode_ell==ell
            if ell>3:
                _mask_nu = (self.mode_nu<self.mode_max_nu)*(self.mode_nu>self.mode_min_nu)
                mask_ell = mask_ell * _mask_nu
            enn_ell = self.mode_enn[mask_ell]
            nu_ell = self.mode_nu[mask_ell]
            fwhm_ell = self.mode_fwhm[mask_ell]
        else:
            enn_ell, nu_ell, fwhm_ell = self.get_gapfilled_nunl(ell)
        num_modes = len(nu_ell)
        ps_list = []

        enn_list = []
        ell_list = []
        emm_list = []
        fwhm_list = []
        nu_list = []

        if visibility_matrix:
            vsm = visibilityMatrix(incl_angle=self.incl_angle)
            LOGGER.info("Using visibility matrix")
            for emm in range(-ell, ell+1):
                _elm = vsm.get_elm(ell, emm)
                for i in range(num_modes):
                    nu_lm = nu_ell[i] + emm*self.rot_rate
                    if self.include_mgfcorr:
                        nu_mgf = self.get_mgfcorr(enn_ell[i], ell)[emm+ell]
                        nu_lm += nu_mgf
                    psl = _elm * self.ps_from_lor(nu_lm, fwhm_ell[i])
                    ps_list.append(psl)#/psl.max())
                    enn_list.append(enn_ell[i])
                    ell_list.append(ell)
                    emm_list.append(emm)
                    nu_list.append(nu_lm)
                    fwhm_list.append(fwhm_ell[i])
        else:
            LOGGER.info("NOT Using visibility matrix")
            for i in range(num_modes):
                psl = self.ps_from_lor(nu_ell[i], fwhm_ell[i])
                ps_list.append(psl)#/psl.max())
        return ps_list, (enn_list, ell_list, emm_list, nu_list, fwhm_list)


    def construct_ps_full(self, nuclist=None, sig1list=None, sig2list=None,
                          visibility_matrix=False, envelope_type='lorentzian'):
        """Constructs the full disk-integrated power spectrum.

        Parameters
        ----------
        nuclist : list or np.ndarray(ndim=1, len=self.lmax+1)
            List of nu_c - peak location of envelope lorentzian
        sig1list : list or np.ndarray(ndim=1, len=self.lmax+1)
            List of sig1 - HWHM of left-half of envelope lorentzian
        sig2list : list or np.ndarray(ndim=1, len=self.lmax+1)
            List of sig2 - HWHM of right-half of envelope lorentzian

        Returns
        -------
        ps_full : np.ndarray(ndim=1)
            Full power spectrum including contributions of all ell upto self.lmax
        """
        if nuclist != None:
            assert len(nuclist)==self.lmax+1, "Number of nu_c is not the same as Lmax"
            assert len(sig1list)==self.lmax+1, "Number of sig1 is not the same as Lmax"
            assert len(sig2list)==self.lmax+1, "Number of sig2 is not the same as Lmax"
        ps_full = 0.0
        
        for ell in range(self.lmax+1):
            _psl = self.construct_ps_ell(ell, visibility_matrix=visibility_matrix)
            if nuclist != None:
                _psl_envelope = self.ps_envelope(nuclist[ell],
                                                 sig1list[ell],
                                                 sig2list[ell])
            else:
                _psl_envelope = self.ps_envelope(type=envelope_type)
            psl = _psl 
            ps_full += psl
        return ps_full * _psl_envelope


    def ps_envelope(self, nuc=3.036e-3, sig1=0.625e-3, sig2=0.942e-3, type='gaussian'):
        """Computes the envelope of the solar power spectrum.
        Stahn (2010) found that using a combination of 2 half-lorentzians,
        centered at nu_c and having different linewidths explains the solar data better.

        Parameters
        ----------
        nu_arr : np.ndarray(ndim=1, dtype=float)
            array containing the set of frequency bins
        nuc : float
            location of lorentzian peak (in mHz)
        sig1 : float
            half-width at half-maximum (HWHM) for the left-half of the lorentzian
        sig2 : float
            half-width at half-maximum (HWHM) for the right-half of the lorentzian
        type : str
            'lorentzian' or 'gaussian'
            Lorentzian option uses 2-half lorentzians as defined by Stahn (2010)
            Gaussian uses the envelope given by Fournier.

        Returns
        -------
        F : np.ndarray(ndim=1, dtype=float)
            envelope for the solar power spectrum.
        """
        if type=='gaussian':
            LOGGER.info("Using gaussian envelope")
            # nu0, sigma0 = 3.5e-3, 0.450e-3 / np.sqrt(8 * np.log(2))
            nu0, sigma0 = 3.0e-3, 2.000e-3 / np.sqrt(8 * np.log(2))

            F = np.exp(-(self.nu_plus - nu0)**2/(2.*sigma0**2))
        elif type=='lorentzian':
            LOGGER.info("Using Lorentzian envelope")
            mask1 = self.nu_plus <= nuc
            mask2 = self.nu_plus > nuc
            F = np.ones_like(self.nu_plus)
            F[mask1] = (1 + ((self.nu_plus[mask1] - nuc)/sig1)**2)**(-2)
            F[mask2] = (1 + ((self.nu_plus[mask2] - nuc)/sig2)**2)**(-2)
        else:
            LOGGER.error(f"Invalid envelope type {type}")
        return F
    
    def bg_apollinaire(self, param, n_harvey=2,):
        '''Compute background model compatible with apollinaire.'''
        def extract_param (param, n_harvey,):
            '''Extract param_harvey, param_gaussian and white noise'''
            assert len(param)==int(n_harvey*3+4), 'Length of parameter vector inconsistent'
            param_harvey = param[:n_harvey*3]
            param_gaussian = param[n_harvey*3:n_harvey*3+3]
            white_noise = param[-1]
            return param_harvey, param_gaussian, white_noise

        def harvey(freq, A, nu_c, alpha):
            '''Compute empirical Harvey law.'''
            num = A
            den = 1. + np.power (freq/nu_c , alpha)
            h = num / den
            return h

        def gauss_env(freq, Hmax, numax, Wenv, asy=0) :
            '''Compute p-modes Gaussian envelope. '''
            return Hmax*np.exp(-np.power((freq-numax)/Wenv, 2))*0.5*(1 + erf((freq-numax)* asy/Wenv))

        model = np.zeros(self.nu_plus.size)
        param_harvey, param_gaussian, noise = extract_param(param, n_harvey)
        param_harvey = np.reshape(param_harvey, (n_harvey, param_harvey.size//n_harvey))
        for elt in param_harvey:
            print(elt)
            model += harvey(self.nu_plus, *elt)
        model += gauss_env(self.nu_plus, *param_gaussian)
        model += noise
        return model

    def get_background_lowfreq(self, type='stahn-nu', A1=1.607, A2=0.542, Ap=1.0, **kwargs):
        """Computes the non-seismic background. There are two different types of background
        functional forms.
        (1) Gaussian -- USE ONLY for testing!
            No justification for the use of this profile. The (mu, sigma) of the
            gaussian profile is obtained by doing a least square fit. The fit is quite bad,
            so it is advisable to use the other option.
        (2) Harvey-like profile -- Stahn (2010) PhD. thesis
            This is a Harvey-like profile, but with an exponent of 4 instead of 2.
            Stahn (2010) claims that this better fits the VIRGO data than the standard
            Harvey profile.

        Parameters
        ----------
        type : str
            Type of function to be used for the non-seismic background

        Returns
        -------
        bg : np.ndarray(ndim=1, dtype=float)
            Non-seismic background for the given set of frequency bins
        """
        if type=='gaussian':
            bg_sig = 7.62e-4
            bg_mu = 2.1545e-3
            bg = 0.0224*np.exp(-(self.nu_plus - bg_mu)**2/(2.*bg_sig**2))
        elif type=='stahn-nu':
            # Note that these fitted values are for unnormalized power spectrum
            # We need an additional scaling factor for use with our normalized power
            # spectra
            noise_photon = 0.004    # ppm^2 muHz^{-1}
            A1 = A1                 # ppm^2 muHz^{-1}
            A2 = A2                 # ppm^2 muHz^{-1]
            t1 = 1390               # seconds
            t2 = 455                # seconds
            bg = (A1/(1 + (t1*self.nu_plus)**4) +
                  A2/(1 + (t2*self.nu_plus)**4) +
                  Ap*noise_photon)
        elif type=='stahn':
            # Note that these fitted values are for unnormalized power spectrum
            # We need an additional scaling factor for use with our normalized power
            # spectra
            noise_photon = 0.004    # ppm^2 muHz^{-1}
            A1 = 1.607              # ppm^2 muHz^{-1}
            A2 = 0.542              # ppm^2 muHz^{-1]
            t1 = 1390               # seconds
            t2 = 455                # seconds
            bg = (A1/(1 + (t1*self.omega_plus)**4) +
                  A2/(1 + (t2*self.omega_plus)**4) +
                  Ap*noise_photon)
        elif type=='bison-paper':
            # Formula from Broomhall, Chaplin, Davies+ (2009) MNRAS, 396, 100-104
            # Values from Kuzslewicz, Davies, Chaplin (2015) EPJ Web Conf. 101, 06041
            noise_photon = 0.004
            sig1 = 0.42*kwargs['sig1f'] # m/s (Granulation)
            sig2 = 0.30*kwargs['sig2f'] # m/s (Mesogranulation)
            sig3 = 5.50*kwargs['sig3f'] # m/s (Supergranulation)
            sig4 = 3.00*kwargs['sig4f'] # m/s (Active regions)
            t1 = 5.e2*kwargs['t1f']   # seconds
            t2 = 1.e4*kwargs['t2f']   # seconds
            t3 = 7.e4*kwargs['t3f']   # seconds
            t4 = 1.e6*kwargs['t4f']   # seconds
            bg = (2*sig1*sig1*t1/(1 + (t1*self.omega_plus)**2) +
                  2*sig2*sig2*t2/(1 + (t2*self.omega_plus)**2) +
                  2*sig3*sig3*t3/(1 + (t3*self.omega_plus)**2) +
                  2*sig4*sig4*t4/(1 + (t4*self.omega_plus)**2) +
                  Ap*noise_photon)
        elif type=="bison-odl":
            sig1 = 0.9
            t1 = 1.0
            gnuc = 3.2e-3
            gsig1 = 4e-4
            gsig2 = 2.0e-3
            ampval = 32.
            g1 = 1e5*np.exp(-(self.omega_plus - gnuc)**2/(2.*gsig1**2))
            g2 = 3.0e4*np.exp(-(self.omega_plus - gnuc + 0.3e-3)**2/(2.*gsig2**2))
            bg = (A1*2*sig1*sig1*t1/(1 + (t1*self.omega_plus)**2)*ampval +
                  A1*2.5e7/(1+1e4*self.omega_plus**0.3) + A2*g1 + A2*g2)
        elif type=='bison-test-2':
            # Formula from Broomhall, Chaplin, Davies+ (2009) MNRAS, 396, 100-104
            # Values from Kuzslewicz, Davies, Chaplin (2015) EPJ Web Conf. 101, 06041
            noise_photon = 0.004
            sig1 = 0.42 # m/s (Granulation)
            sig2 = 0.30 # m/s (Mesogranulation)
            sig3 = 5.50 # m/s (Supergranulation)
            sig4 = 3.00 # m/s (Active regions)
            t1 = 5.e0 # seconds
            t2 = 1.e4 # seconds
            t3 = 7.e4 # seconds
            t4 = 1.e6 # seconds
            print(f"t1 = {t1}")
            bg = (2*sig1*sig1*t1/(1 + (t1*self.omega_plus)**2) +
                  2*sig2*sig2*t2/(1 + (t2*self.omega_plus)**2) +
                  2*sig3*sig3*t3/(1 + (t3*self.omega_plus)**2) +
                  2*sig4*sig4*t4/(1 + (t4*self.omega_plus)**2) +
                  Ap*noise_photon)
        elif type=='apollinaire':
            param = kwargs['param']
            n_harvey = kwargs['n_harvey']
            bg = self.bg_apollinaire(param, n_harvey=n_harvey)
        return bg


    def get_bg_func(self, params=None):
        """Computes the non-seismic background. There are two different types of background
        functional forms.
        (1) Gaussian -- USE ONLY for testing!
            No justification for the use of this profile. The (mu, sigma) of the
            gaussian profile is obtained by doing a least square fit. The fit is quite bad,
            so it is advisable to use the other option.
        (2) Harvey-like profile -- Stahn (2010) PhD. thesis
            This is a Harvey-like profile, but with an exponent of 4 instead of 2.
            Stahn (2010) claims that this better fits the VIRGO data than the standard
            Harvey profile.

        This function is used for fitting the background function.

        Parameters
        ----------
        A1 : float
            amplitude of first Harvey-like profile

        A2 : float
            amplitude of second Harvey-like profile

        Returns
        -------
        bg : np.ndarray(ndim=1, dtype=float)
            Non-seismic background for the given set of frequency bins
        """
        # Note that these fitted values are for unnormalized power spectrum
        # We need an additional scaling factor for use with our normalized power
        # spectra
        if params[0]==None:
            A1 = 1.607
            A2 = 0.542
            t1 = 1390/3600.
            t2 = 455/3600.
            noise_p = 0.4
        else:
            A1 = params[0]
            A2 = params[1]
            t1 = params[2]
            t2 = params[3]
            noise_p = params[4]
        noise_photon = noise_p/100.    # ppm^2 muHz^{-1}
        A1 = A1                 # ppm^2 muHz^{-1}
        A2 = A2                 # ppm^2 muHz^{-1}
        t1 = t1*3600.           # seconds
        t2 = t2*3600.           # seconds
        bg = (A1/(1 + (t1*self.nu_plus)**4) +
              A2/(1 + (t2*self.nu_plus)**4) +
              0.0*noise_photon)
        return bg


    def get_background_midfreq(self, jmax=3, factors=None):
        if factors == None: factors = np.ones(jmax+1)
        assert len(factors) == jmax+1, "Number of factors doesn't match polynomial order jmax"
        nuplus_mhz = self.nu_plus * 1e3

        p = []
        for j in range(jmax+1):
            facs = np.zeros(jmax+1)
            facs[-j-1] = 1.0
            _poly = np.poly1d(facs)(nuplus_mhz) * self.ps_envelope(type='gaussian')
            p.append(_poly)
        return p


    def get_fit1d_arrays(self):
        bgm = get_background_midfreq(self)
        ps_ell = []
        for ell in range(self.lmax+1):
            ps_ell.append(self.construct_ps_ell(ell=ell))



class visibilityMatrix(stellarPS):
    """Class to generate visibility-matrix for computing change in mode-visibility
    due to inclination angle."""
    __attributes__ = ["cosi"]
    __methods__ = ["get_elm",
                   "get_elm_gs"]

    def __init__(self, incl_angle=0., degrees=False):
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
        lpmf = np.math.factorial(lpm)
        lmmf = np.math.factorial(lmm)
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


