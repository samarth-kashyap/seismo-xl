import h5py
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Local imports
from src.globalvars import globalVars
from src.stellarspec import stellarPS

# Defining some global variables
GVARS = globalVars()

#--------------------- ARGUMENT PARSER ---------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--source', type=str, default='valeriy', help='VIRGO data source')
parser.add_argument('--channel', type=str, default='blue', help='VIRGO channel (default=blue)')
parser.add_argument('--skipmax', type=int, default=40, help='Maximum skip number (default=40)')
parser.add_argument('--Ncarr', type=int, default=3, help='Number of Carrington rotations')
parser.add_argument('--lmax', type=np.int32, default=3, help='Lmax (default=3)')
parser.add_argument('--inclang', type=np.int32, default=90, help='Inclination angle')
parser.add_argument('--ndays', type=float, default=72, help='Number of observation days')
parser.add_argument('--realizations', type=np.int32, default=1000, help='Realizations for MonteCarlo')
parser.add_argument('--freqmin', type=float, default=0.5, help='Minimum freq in mHz')
parser.add_argument('--freqmax', type=float, default=5.5, help='Maximum freq in mHz')
parser.add_argument('--scratch-dir', type=str, default='/scratch/seismo/kashyap/processed/sun-intg',
                    help='Base directory for data and fits (default: /scratch/.../sun-intg)')
parser.add_argument('--obs-dir', type=str, default=None,
                    help='Directory with Larson_Schou_MDI_2015.dat reference data')
ARGS = parser.parse_args()
#----------------------------------------------------------------------------

assert ARGS.freqmin>0. and ARGS.freqmax<10., "Min freq out of range"
assert ARGS.freqmax>0. and ARGS.freqmax<10., "Max freq out of range"
assert ARGS.freqmax>ARGS.freqmin, "maxfreq < minfreq; exiting"


scratch_dir = ARGS.scratch_dir

# Load reference mode parameters
ELLS, ENNS, NUS, FWHMS, SIG_FWHMS = GVARS.load_data(obs_dir=ARGS.obs_dir)
data_dir = f"{scratch_dir}/data/{ARGS.source}-{ARGS.channel}-Ncarr{ARGS.Ncarr}-skip{ARGS.skipmax:02d}"
fits_dir = f"{scratch_dir}/ps-fits/{ARGS.source}-{ARGS.channel}-Ncarr{ARGS.Ncarr}-skip{ARGS.skipmax:02d}-ell{ARGS.lmax}-i{ARGS.inclang:02d}"


def filter_butterworth_bandpass(_f1, tt1, forder=12,):
    """Apply a low-pass Butterworth filter to a time series.

    Parameters
    ----------
    _f1 : np.ndarray, shape (N,)
        Frequency array corresponding to ``tt1`` in Hz.
    tt1 : np.ndarray, shape (N,)
        Input time series (or spectrum) to be filtered.
    forder : int, optional
        Order of the Butterworth filter.

    Returns
    -------
    tt1_filtered : np.ndarray, shape (N,)
        Filtered version of ``tt1``.
    """
    freqmin = ARGS.freqmin*1e-3
    freqmax = ARGS.freqmax*1e-3
    b, a = signal.butter(forder, 1./(period_min_yr*365.*24.*3600.), 'low', analog=True)
    w, h = signal.freqs(b, a, worN=_f1)
    tt1_filtered = np.fft.irfft(abs(h)*np.fft.rfft(tt1))
    return tt1_filtered


def get_freqlags(refarr, pfilt_list, maxlag=20):
    """Compute frequency lags via cross-correlation and Gaussian fitting.

    Parameters
    ----------
    refarr : np.ndarray, shape (N,)
        Reference (observed) power spectrum.
    pfilt_list : np.ndarray, shape (4, N)
        Matched-filter power spectra for harmonic degrees 0-3.
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.

    Returns
    -------
    corr_mats : tuple of np.ndarray
        ``(corr_mat, corr_mat_gauss)`` - raw and Gaussian-fitted correlation
        matrices of shape ``(4, 2*maxlag+1)``.
    corr_args : tuple of np.ndarray
        ``(corr_matarg, corr_matarg_gauss)`` - peak-lag indices from argmax
        and from the Gaussian fit, each of shape ``(4,)``.
    """
    corr_mat = np.zeros((4, 2*maxlag+1))
    corr_mat_gauss = np.zeros((4, 2*maxlag+1))
    corr_matarg = np.zeros(4)
    corr_matarg_gauss = np.zeros(4)
    
    lags_list, corr_list, corrnlist, corrbg_list = [], [], [], []
    for jdx in range(pfilt_list.shape[0]):
        p0 = [1., 0., 1., 0.]
        lags, corr = compute_cc(pfilt_list[jdx]/1e10, refarr/1e10, maxlag=maxlag)
        try:
            coeff, var_matrix = curve_fit(gaussian, lags, corr, p0=p0)
        except RuntimeError:
            continue
        corr_mat_gauss[jdx, :] = gaussian(lags, *coeff)
        corr_mat[jdx, :] = corr
        max_idx = np.argmax(corr)
        corr_matarg[jdx] = lags[max_idx]
        corr_matarg_gauss[jdx] = coeff[1]
    return (corr_mat, corr_mat_gauss), (corr_matarg, corr_matarg_gauss)


def get_freqlags_corrected(refarr, pfilt_list, pexcl_list, maxlag=20):
    """Compute background-corrected frequency lags via cross-correlation.

    The leakage-correction (excluded-mode) cross-correlation is subtracted
    from the filter cross-correlation before fitting with a Gaussian.

    Parameters
    ----------
    refarr : np.ndarray, shape (N,)
        Reference (observed) power spectrum.
    pfilt_list : np.ndarray, shape (4, N)
        Matched-filter power spectra for harmonic degrees 0-3.
    pexcl_list : np.ndarray, shape (4, N)
        Leakage-correction (excluded-mode) spectra for harmonic degrees 0-3.
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.

    Returns
    -------
    corr_mats : tuple of np.ndarray
        ``(corr_mat, corr_mat_gauss)`` - background-corrected and
        Gaussian-fitted correlation matrices of shape ``(4, 2*maxlag+1)``.
    corr_args : tuple of np.ndarray
        ``(corr_matarg, corr_matarg_gauss)`` - peak-lag indices from argmax
        and from the Gaussian fit, each of shape ``(4,)``.
    corrbg_list : list of np.ndarray
        Background cross-correlation for each harmonic degree.
    """
    # print(f"max frequency lag = {maxlag*dfreq:.2f} muHz")
    # corr_mat stores the correlation matrix [ell, time_chunk, lag]
    # corr_mat_gauss stores the gaussian fit [ell, time_chunk, lag]
    # corr_matarg stores the index corresponding to maximum corr [ell, time_chunk]
    # corr_matarg_gauss max corr for the gaussian fit [ell, time_chunk]

    # things to check
    # the leakage correction filter needs to be applied on pexcl_list or pfilt_list?
    # this can significantly change things.
    
    corr_mat = np.zeros((4, 2*maxlag+1))
    corr_mat_gauss = np.zeros((4, 2*maxlag+1))
    corr_matarg = np.zeros(4)
    corr_matarg_gauss = np.zeros(4)
    
    lags_list, corr_list, corrnlist, corrbg_list = [], [], [], []
    for jdx in range(pfilt_list.shape[0]):
        p0 = [1., 0., 1., 0.]
        lags, corr = compute_cc(pfilt_list[jdx]/1e10, refarr/1e10, maxlag=maxlag)
        # lags_bg, corr_bg = compute_cc(pfilt_list[jdx]/1e10, pexcl_list[jdx]/1e10, maxlag=maxlag)
        lags_bg, corr_bg = compute_cc(pexcl_list[jdx]/1e10, refarr/1e10, maxlag=maxlag)
        corr = corr - corr_bg
        coeff = [0, 0, 0, 0]
        try:
            coeff, var_matrix = curve_fit(gaussian, lags, corr, p0=p0)
        except RuntimeError:
            coeff[1] = np.nan
        corr_mat_gauss[jdx, :] = gaussian(lags, *coeff)
        corr_mat[jdx, :] = corr
        max_idx = np.argmax(corr)
        corr_matarg[jdx] = lags[max_idx]
        corr_matarg_gauss[jdx] = coeff[1]
        corrbg_list.append(corr_bg)
    return (corr_mat, corr_mat_gauss), (corr_matarg, corr_matarg_gauss), corrbg_list


def gaussian_gfilt(x, mu, fwhm):
    """Evaluate a Gaussian profile given a mean and FWHM.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Domain over which the Gaussian is evaluated.
    mu : float
        Location of the Gaussian peak.
    fwhm : float
        Full width at half maximum (same units as ``x``).

    Returns
    -------
    np.ndarray, shape (N,)
        Gaussian profile with unit peak amplitude.
    """
    sigma = fwhm / np.sqrt(8. * np.log(2.))
    return np.exp(-(x-mu)**2/2./sigma/sigma)


def get_pslbg(SPS, visibility_matrix=True, return_nl_list=True):
    """Construct the list of power-spectrum components for the model fit.

    The total power spectrum is modelled as a linear combination of:

    1. Lorentzians for each (n, ell, m) mode.
    2. Harvey-like profiles for the granulation background.
    3. Photon noise (white noise floor).

    Parameters
    ----------
    SPS : stellarPS
        Configured instance of :class:`src.stellarspec.stellarPS`.
    visibility_matrix : bool, optional
        If ``True``, apply geometrical visibility corrections to mode heights.
    return_nl_list : bool, optional
        If ``True``, also return per-mode metadata lists.

    Returns
    -------
    psdict : tuple
        ``(psl_bg, ps_nlm_dict)`` where ``psl_bg`` is an ``np.ndarray`` of
        shape ``(n_components, N_freq)`` and ``ps_nlm_dict`` is a dict
        mapping ``'ell-enn'`` keys to per-m Lorentzian lists.
    fmhz : np.ndarray, shape (N_freq,)
        Frequency bins in mHz.
    enn_list : list of int
        Radial orders (only when ``return_nl_list`` is ``True``).
    ell_list : list of int
        Harmonic degrees (only when ``return_nl_list`` is ``True``).
    nu_list : list of float
        Mode frequencies in Hz (only when ``return_nl_list`` is ``True``).
    gamma_list : list of float
        Mode line-widths in Hz (only when ``return_nl_list`` is ``True``).
    """
    fmhz = SPS.nu_plus * 1e3

    # Defining the background Harvey-like profiles
    bgl = []
    psl_ells = []
    psl_nlm = []
    ell_list = []
    enn_list = []
    nu_list = []
    gamma_list = []
    ps_nlm_dict = {}
    if ARGS.source=="bison":
        bgtype = "bison"
    else:
        bgtype = "stahn-nu"
    bgl.append(SPS.get_background_lowfreq(A1=1.0, A2=0.0, Ap=0.0, type=bgtype))
    bgl.append(SPS.get_background_lowfreq(A1=0.0, A2=1.0, Ap=0.0, type=bgtype))

    for ell in range(SPS.lmax+1):
        psl, enns, ells, nus, gammas = SPS.construct_ps_list(ell=ell,
                                                             visibility_matrix=visibility_matrix,
                                                             return_nl_list=return_nl_list)
        print(f"Num modes [ell={ell:d}] = {len(ells)}")
        psl_ells = [*psl_ells, *psl[0]]
        psl_nlm = [*psl_nlm, *psl[1]]
        ell_list = [*ell_list, *ells]
        enn_list = [*enn_list, *enns]
        nu_list = [*nu_list, *nus]
        gamma_list = [*gamma_list, *gammas]

        for _enn in enns:
            _psnlm = SPS.construct_ps_normed_nlm(enn=_enn, ell=ell, shiftfreq=0.,
                                                 scalefwhm=1., stahn=True)
            ps_nlm_dict[f"{ell:d}-{_enn:02d}"] = _psnlm
            

    psl_bg = [*psl_ells, *bgl]
    psl_nlm = [*psl_nlm, *bgl]
    ps_nlm_dict["bg1"] = bgl[0]
    ps_nlm_dict["bg2"] = bgl[1]
    psl_bg = np.array(psl_bg)
    if return_nl_list:
        return (psl_bg, ps_nlm_dict), fmhz, enn_list, ell_list, nu_list, gamma_list
    else:
        return psl_bg, fmhz


def noisify(iparr):
    """Apply chi-squared (2 d.o.f.) noise to a power spectrum.

    Each frequency bin is multiplied by an independent draw from a
    chi-squared distribution with 2 degrees of freedom, as appropriate for
    an exponentially distributed power spectrum.

    Parameters
    ----------
    iparr : np.ndarray
        Input (model) power spectrum.

    Returns
    -------
    noisy_arr : np.ndarray
        Noise realisation of the input spectrum, same shape as ``iparr``.
    """
    noise = np.random.randn(*(iparr.shape))**2
    noisy_arr = iparr*noise
    return noisy_arr


def gaussian(x, *p):
    """Evaluate a Gaussian with an additive DC offset.

    Convenience wrapper for use with ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    x : np.ndarray
        Domain on which the Gaussian is evaluated.
    *p : float
        Four parameters ``(A, mu, sigma, k)`` where

        * ``A``     - amplitude,
        * ``mu``    - centroid,
        * ``sigma`` - standard deviation,
        * ``k``     - DC offset.

    Returns
    -------
    np.ndarray
        Gaussian profile on ``x``.
    """
    A, mu, sigma, k = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + k


def compute_cc(arr1, arr2, maxlag=20):
    """Compute the discrete cross-correlation for lags in ``[-maxlag, maxlag]``.

    Parameters
    ----------
    arr1 : np.ndarray, shape (N,)
        First signal (e.g. the raw observed power spectrum).
    arr2 : np.ndarray, shape (N,)
        Second signal (e.g. the model filter power spectrum).
    maxlag : int, optional
        Maximum lag index. The returned lag array spans
        ``np.arange(-maxlag, maxlag+1)``.

    Returns
    -------
    lags : np.ndarray of int, shape (2*maxlag+1,)
        Array of integer lag indices.
    cc : np.ndarray of float, shape (2*maxlag+1,)
        Cross-correlation values at each lag.
    """
    padded1arr = np.pad(arr1, (maxlag+1, maxlag+1), 'constant', constant_values=(0, 0))
    padded2arr = np.pad(arr2, (maxlag+1, maxlag+1), 'constant', constant_values=(0, 0))
    maxlag = abs(int(maxlag))
    assert maxlag > 0, "maxlag should be at least 1"
    if maxlag%2>0: maxlag += 1
    cc = np.zeros(2*maxlag+1)
    lags = np.arange(-maxlag, maxlag+1)
    for idx in range(len(cc)):
        cc[idx] = np.sum(padded1arr*np.roll(padded2arr, idx-maxlag))
    return lags, cc


def compute_delnu(pmod_list, pfilt_list, pexcl_list, corrected=False):
    """Estimate frequency shifts via Monte Carlo for each harmonic degree.

    Generates ``ARGS.realizations`` noise realisations of each per-ell model
    spectrum and computes the cross-correlation peak position for each,
    collecting the Gaussian-fit centroid as the frequency-shift estimate.

    Parameters
    ----------
    pmod_list : list of np.ndarray
        Per-ell model power spectra ``[pmod0, pmod1, pmod2, pmod3]``.
    pfilt_list : np.ndarray, shape (4, N)
        Matched-filter spectra for harmonic degrees 0-3.
    pexcl_list : np.ndarray, shape (4, N)
        Leakage-correction spectra for harmonic degrees 0-3.
    corrected : bool, optional
        If ``True``, apply the background (leakage) correction via
        :func:`get_freqlags_corrected`; otherwise use :func:`get_freqlags`.

    Returns
    -------
    collect0fitval, collect1fitval, collect2fitval, collect3fitval : np.ndarray
        Arrays of Gaussian-fit centroids (in lag-index units) for each of
        the four harmonic degrees across all Monte Carlo realisations.
    """
    collect1fitval = []
    collect2fitval = []
    collect3fitval = []
    for idx in tqdm(range(ARGS.realizations), desc='performing Montecarlo'):
        if corrected:
            __a0, __a00, cbg0 = get_freqlags_corrected(noisify(pmod_list[0]), pfilt_list, pexcl_list)
            __a1, __a10, cbg1 = get_freqlags_corrected(noisify(pmod_list[1]), pfilt_list, pexcl_list)
            __a2, __a20, cbg2 = get_freqlags_corrected(noisify(pmod_list[2]), pfilt_list, pexcl_list)
            __a3, __a30, cbg3 = get_freqlags_corrected(noisify(pmod_list[3]), pfilt_list, pexcl_list)
            collect0fitval.append(__a00[1][0])
            collect1fitval.append(__a10[1][1])
            collect2fitval.append(__a20[1][2])
            collect3fitval.append(__a30[1][3])
        else:
            __a0, __a00 = get_freqlags(noisify(pmod_list[0]), pfilt_list)
            __a1, __a10 = get_freqlags(noisify(pmod_list[1]), pfilt_list)
            __a2, __a20 = get_freqlags(noisify(pmod_list[2]), pfilt_list)
            __a3, __a30 = get_freqlags(noisify(pmod_list[3]), pfilt_list)
            collect0fitval.append(__a00[1][0])
            collect1fitval.append(__a10[1][1])
            collect2fitval.append(__a20[1][2])
            collect3fitval.append(__a30[1][3])

    collect0fitval = np.array(collect0fitval)
    collect1fitval = np.array(collect1fitval)
    collect2fitval = np.array(collect2fitval)
    collect3fitval = np.array(collect3fitval)
    return collect0fitval, collect1fitval, collect2fitval, collect3fitval



def testfunc():
    """Run a quick Monte Carlo test using module-level shifted spectra.

    Uses the globally-defined ``pmod0s``-``pmod3s`` (shifted model spectra)
    and ``pfilt_list`` to compute Gaussian-fit frequency-shift centroids for
    ``ARGS.realizations`` noise realisations, then plots histograms.

    Returns
    -------
    collect0fitval, collect1fitval, collect2fitval, collect3fitval : np.ndarray
        Gaussian-fit centroids (lag-index units) for harmonic degrees 0-3.
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes
    """
    collect0fitval = []
    collect1fitval = []
    collect2fitval = []
    collect3fitval = []
    for idx in tqdm(range(ARGS.realizations), desc='performing Montecarlo'):
        __a0, __a00 = get_freqlags(noisify(pmod0s), pfilt_list)
        __a1, __a10 = get_freqlags(noisify(pmod1s), pfilt_list)
        __a2, __a20 = get_freqlags(noisify(pmod2s), pfilt_list)
        __a3, __a30 = get_freqlags(noisify(pmod3s), pfilt_list)
        collect0fitval.append(__a00[1][0])
        collect1fitval.append(__a10[1][1])
        collect2fitval.append(__a20[1][2])
        collect3fitval.append(__a30[1][3])

    collect0fitval = np.array(collect0fitval)
    collect1fitval = np.array(collect1fitval)
    collect2fitval = np.array(collect2fitval)
    collect3fitval = np.array(collect3fitval)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    axs.hist((collect0fitval + shiftval)*dfreq,
             histtype=u'step', label='ell=0', bins=np.linspace(-2, 2, 100))
    axs.hist((collect1fitval + shiftval)*dfreq,
             histtype=u'step', label='ell=1', bins=np.linspace(-2, 2, 100))
    axs.hist((collect2fitval + shiftval)*dfreq,
             histtype=u'step', label='ell=2', bins=np.linspace(-2, 2, 100))
    axs.hist((collect3fitval + shiftval)*dfreq,
             histtype=u'step', label='ell=3', bins=np.linspace(-2, 2, 100))
    fig.supxlabel('$\\delta\\nu^\\mathrm{pred} - \\delta\\nu^\\mathrm{true}$ in $\\mu$Hz',
                  fontsize=12)
    axs.legend()
    fig.tight_layout()
    return collect0fitval, collect1fitval, collect2fitval, collect3fitval, fig, axs



if __name__ == "__main__":
    ## TODO LIST
    # [x]  amplitudes and lorentzians for different ell
    # [x]  amplitudes and lorentizians for different (ell, emm) combos
    # [x]  load gfilter
    # [x]  separate background fitting
    # [x]  add frequency filtering -- better to add a butterworth filter,
    #      rather than just filtering out a part of the spectrum
    #      - [x] basic filtering (rectangular filter)
    #      - [x] smooth filtering (butterworth filter)
    #      - [x] cross-correlation should be fine as long as 0-padding is done
    # [x] integrate fitting using peak-bagged frequencies
    # [x] realistic estimation of errors in delnu -- notebook created

    ell = np.load(f'{fits_dir}/fitted-ell-list.npy')
    enn = np.load(f'{fits_dir}/fitted-enn-list.npy')
    nus = np.load(f'{fits_dir}/fitted-nu-list-mod.npy')
    fwhm = np.load(f'{fits_dir}/fitted-fwhm-list-mod.npy')
    pobs = np.load(f'{data_dir}/psref.npy')
    amps = np.load(f'{fits_dir}/fitted-mode-amplitudes.npy')
    amps_llk = np.load(f'{fits_dir}/fitted-mode-amplitudes-mod.npy')
    bgamps = np.load(f'{fits_dir}/fitted-mode-amplitudes-mod.npy')[-2:]
    kth = np.load(f'{fits_dir}/numean-kernels.npy')
    years = np.load(f'{data_dir}/years.npy')
    bsp = np.load(f'{scratch_dir}/bsp-basis/bsp_knotnum_15.npy')
    amps_llk[-2:] = bgamps
    amps_llk = amps*1.0

    SPS = stellarPS(lmax=ARGS.lmax,
                  mode_ell=ELLS,
                  mode_enn=ENNS,
                  mode_nu=NUS,
                  mode_fwhm=FWHMS,
                  mode_sigfwhm=SIG_FWHMS,
                  incl_angle=ARGS.inclang*np.pi/180.,
                  cadence=60.,
                  obs_ndays=ARGS.ndays*1.0,)
    freq_arr = SPS.nu_plus
    freq_mhz = freq_arr*1e3
    MASK_FREQ = (freq_mhz>=ARGS.freqmin)*(freq_mhz<=ARGS.freqmax)

    freq_arr = freq_arr[MASK_FREQ]
    freq_mhz = freq_mhz[MASK_FREQ]

    gfilter = gaussian_gfilt(freq_arr*1e3, 3.0, 2.0)
    print(f'LOADING SUCESS')
    psdict, fmhz, enn_list, ell_list, nu_list, gamma_list = get_pslbg(SPS, return_nl_list=True)
    psl_bg, ps_nlm_dict = psdict
    psl_bg = psl_bg[:, MASK_FREQ]

    assert np.prod(ell_list==ell), "Loaded amplitudes for ell dont match the current ell_list"
    assert np.prod(enn_list==enn), "Loaded amplitudes for enn dont match the current enn_list"

    mask0 = np.array(ell_list)==0
    mask1 = np.array(ell_list)==1
    mask2 = np.array(ell_list)==2
    mask3 = np.array(ell_list)==3

    bgfit = amps_llk[-2:] @ psl_bg[-2:]
    psfit = amps_llk @ psl_bg

    plt.figure()
    plt.plot(freq_arr*1e3, psfit)
    plt.xlabel('Frequency in mHz')
    print(f"----Number of frequency bins = {len(freq_arr)}")

    # Computing the filtered power spectra
    pmod0 = np.squeeze((amps_llk[:-2][mask0] @ psl_bg[:-2][mask0, :] + bgfit)*gfilter)
    pmod1 = np.squeeze((amps_llk[:-2][mask1] @ psl_bg[:-2][mask1, :] + bgfit)*gfilter)
    pmod2 = np.squeeze((amps_llk[:-2][mask2] @ psl_bg[:-2][mask2, :] + bgfit)*gfilter)
    pmod3 = np.squeeze((amps_llk[:-2][mask3] @ psl_bg[:-2][mask3, :] + bgfit)*gfilter)

    # Computing shifted power spectra
    shiftval_list = [5, 7, 10, 15, 20]
    shiftval = -shiftval_list[1]
    dfreq = (freq_mhz[1] - freq_mhz[0])*1e3
    print(f"Defined delnu = {shiftval*dfreq:.2f} muHz")
    pmod0s = np.squeeze((amps_llk[:-2][mask0] @ np.roll(psl_bg[:-2][mask0, :], shiftval, axis=1) + bgfit)*gfilter)
    pmod1s = np.squeeze((amps_llk[:-2][mask1] @ np.roll(psl_bg[:-2][mask1, :], shiftval, axis=1) + bgfit)*gfilter)
    pmod2s = np.squeeze((amps_llk[:-2][mask2] @ np.roll(psl_bg[:-2][mask2, :], shiftval, axis=1) + bgfit)*gfilter)
    pmod3s = np.squeeze((amps_llk[:-2][mask3] @ np.roll(psl_bg[:-2][mask3, :], shiftval, axis=1) + bgfit)*gfilter)

    pmod0123s = (np.squeeze(amps_llk[:-2][mask0] @ np.roll(psl_bg[:-2][mask0, :], shiftval, axis=1)) +
                np.squeeze(amps_llk[:-2][mask1] @ np.roll(psl_bg[:-2][mask1, :], shiftval, axis=1)) +
                np.squeeze(amps_llk[:-2][mask2] @ np.roll(psl_bg[:-2][mask2, :], shiftval, axis=1)) +
                np.squeeze(amps_llk[:-2][mask3] @ np.roll(psl_bg[:-2][mask3, :], shiftval, axis=1)) + bgfit)*gfilter

    pexc0 = np.squeeze((psfit - amps_llk[:-2][mask0] @ psl_bg[:-2][mask0, :] - bgfit)*gfilter)
    pexc1 = np.squeeze((psfit - amps_llk[:-2][mask1] @ psl_bg[:-2][mask1, :] - bgfit)*gfilter)
    pexc2 = np.squeeze((psfit - amps_llk[:-2][mask2] @ psl_bg[:-2][mask2, :] - bgfit)*gfilter)
    pexc3 = np.squeeze((psfit - amps_llk[:-2][mask3] @ psl_bg[:-2][mask3, :] - bgfit)*gfilter)

    # pexc0 = psfit*gfilter - pmod0
    # pexc1 = psfit*gfilter - pmod1
    # pexc2 = psfit*gfilter - pmod2
    # pexc3 = psfit*gfilter - pmod3

    pfilt_list = [pmod0, pmod1, pmod2, pmod3]
    pmods_list = [pmod0s, pmod1s, pmod2s, pmod3s]
    pexcl_list = [pexc0, pexc1, pexc2, pexc3]
    pfilt_list = np.array(pfilt_list)
    pexcl_list = np.array(pexcl_list)

    c0f, c1f, c2f, c3f, fig, axs = testfunc()
    fig.show()

    tfile = h5py.File('data/test.h5')


    sys.exit()

    coll0123 = compute_delnu(pmods_list, pfilt_list, pexcl_list, corrected=False)
    coll0123_corr = compute_delnu(pmods_list, pfilt_list, pexcl_list, corrected=True)

    collect0fitval = coll0123[0]
    collect1fitval = coll0123[1]
    collect2fitval = coll0123[2]
    collect3fitval = coll0123[3]
    collect0fitval_corr = coll0123_corr[0]
    collect1fitval_corr = coll0123_corr[1]
    collect2fitval_corr = coll0123_corr[2]
    collect3fitval_corr = coll0123_corr[3]

    collect_fitval = []
    for idx in tqdm(range(ARGS.realizations), desc='MonteCarlo combined'):
        __a0, __a00 = get_freqlags(noisify(pmod0123s), np.array([psfit*gfilter,]))
        collect_fitval.append(__a00[1][0])
    collect_fitval = np.array(collect_fitval)
    print(f"===============================================================")
    print(f" freqmin = {ARGS.freqmin:.2f}; freqmax = {ARGS.freqmax:.2f}")
    print(f"===============================================================")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    axs[0].hist((collect0fitval + shiftval)*dfreq, histtype=u'step', label='ell=0', bins=np.linspace(-2, 2, 100), density=True)
    axs[0].hist((collect1fitval + shiftval)*dfreq, histtype=u'step', label='ell=1', bins=np.linspace(-2, 2, 100), density=True)
    axs[0].hist((collect2fitval + shiftval)*dfreq, histtype=u'step', label='ell=2', bins=np.linspace(-2, 2, 100), density=True)
    axs[0].hist((collect3fitval + shiftval)*dfreq, histtype=u'step', label='ell=3', bins=np.linspace(-2, 2, 100), density=True)
    axs[0].set_title('Before pexcl correction')

    axs[1].hist((collect0fitval_corr + shiftval)*dfreq, histtype=u'step',
                label='ell=0', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].hist((collect1fitval_corr + shiftval)*dfreq, histtype=u'step',
                label='ell=1', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].hist((collect2fitval_corr + shiftval)*dfreq, histtype=u'step',
                label='ell=2', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].hist((collect3fitval_corr + shiftval)*dfreq, histtype=u'step',
                label='ell=3', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].set_title('After pexcl correction')
    fig.supxlabel('$\\delta\\nu^\\mathrm{pred} - \\delta\\nu^\\mathrm{true}$ in $\\mu$Hz', fontsize=12)
    for _axs in axs: _axs.legend()
    fig.tight_layout()

    sys.exit()
    
    # Computing delnu with correction factor
    collect0fitval_corr = []
    collect1fitval_corr = []
    collect2fitval_corr = []
    collect3fitval_corr = []
    for idx in tqdm(range(ARGS.realizations), desc='performing Montecarlo'):
        __a0, __a00, cbg0 = get_freqlags_corrected(noisify(pmod0s), pfilt_list, pexcl_list)
        __a1, __a10, cbg1 = get_freqlags_corrected(noisify(pmod1s), pfilt_list, pexcl_list)
        __a2, __a20, cbg2 = get_freqlags_corrected(noisify(pmod2s), pfilt_list, pexcl_list)
        __a3, __a30, cbg3 = get_freqlags_corrected(noisify(pmod3s), pfilt_list, pexcl_list)
        collect0fitval_corr.append(__a00[1][0])
        collect1fitval_corr.append(__a10[1][1])
        collect2fitval_corr.append(__a20[1][2])
        collect3fitval_corr.append(__a30[1][3])

    collect0fitval_corr = np.array(collect0fitval_corr)
    collect1fitval_corr = np.array(collect1fitval_corr)
    collect2fitval_corr = np.array(collect2fitval_corr)
    collect3fitval_corr = np.array(collect3fitval_corr)

    # Computing delnu without correction factor
    collect0fitval = []
    collect1fitval = []
    collect2fitval = []
    collect3fitval = []
    for idx in tqdm(range(ARGS.realizations), desc='performing Montecarlo'):
        __a0, __a00 = get_freqlags(noisify(pmod0s), pfilt_list)
        __a1, __a10 = get_freqlags(noisify(pmod1s), pfilt_list)
        __a2, __a20 = get_freqlags(noisify(pmod2s), pfilt_list)
        __a3, __a30 = get_freqlags(noisify(pmod3s), pfilt_list)
        collect0fitval.append(__a00[1][0])
        collect1fitval.append(__a10[1][1])
        collect2fitval.append(__a20[1][2])
        collect3fitval.append(__a30[1][3])

    collect0fitval = np.array(collect0fitval)
    collect1fitval = np.array(collect1fitval)
    collect2fitval = np.array(collect2fitval)
    collect3fitval = np.array(collect3fitval)
