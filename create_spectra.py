import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# Local imports
from src.globalvars import globalVars
from src.solarspec import solarPS

# Defining some global variables
GVARS = globalVars()
ELLS, ENNS, NUS, FWHMS, SIG_FWHMS = GVARS.load_data()


#--------------------- ARGUMENT PARSER ---------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--source', type=str, default='valeriy', help='VIRGO data source')
parser.add_argument('--channel', type=str, default='blue', help='VIRGO channel (default=blue)')
parser.add_argument('--skipmax', type=int, default=40, help='Maximum skip number (default=40)')
parser.add_argument('--Ncarr', type=int, default=3, help='Number of Carrington rotations')
parser.add_argument('--lmax', type=np.int32, default=3, help='Lmax (default=3)')
parser.add_argument('--inclang', type=np.int32, default=90, help='Inclination angle')
parser.add_argument('--ndays', type=np.int32, default=72, help='Number of observation days')
parser.add_argument('--realizations', type=np.int32, default=1000, help='Realizations for MonteCarlo')
parser.add_argument('--freqmin', type=float, default=0.5, help='Minimum freq in mHz')
parser.add_argument('--freqmax', type=float, default=5.5, help='Maximum freq in mHz')
ARGS = parser.parse_args()
#----------------------------------------------------------------------------

assert ARGS.freqmin>0. and ARGS.freqmax<6., "Min freq out of range"
assert ARGS.freqmax>0. and ARGS.freqmax<6., "Max freq out of range"
assert ARGS.freqmax>ARGS.freqmin, "maxfreq < minfreq; exiting"


scratch_dir = f"/scratch/seismo/kashyap/processed/sun-intg"
data_dir = f"{scratch_dir}/data/{ARGS.source}-{ARGS.channel}-Ncarr{ARGS.Ncarr}-skip{ARGS.skipmax:02d}"
fits_dir = f"{scratch_dir}/ps-fits/{ARGS.source}-{ARGS.channel}-Ncarr{ARGS.Ncarr}-skip{ARGS.skipmax:02d}-ell{ARGS.lmax}-i{ARGS.inclang:02d}"


def filter_butterworth_bandpass(_f1, tt1, forder=12,):
    """Applying butterworth filter to the observed spectra.

    Parameters
    ----------
    :_f1: Frequency array
    :type: np.ndarray(ndim=1, dtype=np.float64)

    :tt1: Travel-time array
    :type: np.ndarray(ndim=1, dtype=np.float64)

    Returns
    -------
    :tt1_filtered: Filtered travel-time array
    :type: np.ndarray(ndim=1, dtype=np.float64)
    """
    freqmin = ARGS.freqmin*1e-3
    freqmax = ARGS.freqmax*1e-3
    b, a = signal.butter(forder, 1./(period_min_yr*365.*24.*3600.), 'low', analog=True)
    w, h = signal.freqs(b, a, worN=_f1)
    tt1_filtered = np.fft.irfft(abs(h)*np.fft.rfft(tt1))
    return tt1_filtered



def get_freqlags(refarr, pfilt_list, maxlag=20):
    # print(f"max frequency lag = {maxlag*dfreq:.2f} muHz")
    
    # corr_mat stores the correlation matrix [ell, time_chunk, lag]
    # corr_mat_gauss stores the gaussian fit [ell, time_chunk, lag]
    # corr_matarg stores the index corresponding to maximum corr [ell, time_chunk]
    # corr_matarg_gauss max corr for the gaussian fit [ell, time_chunk]
    
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
    """Compute frequecy lags

    Parameters
    ----------
    :refarr: Observed power spectrum
    :type: np.ndarray(ndim=1, dtype=float)[freq]

    :pfilt_list: List of MI filters for different ell
    :type: np.ndarray(ndim=2, dtype=float)[ell, freq]

    :pexcl_list: List of LC filters for different ell
    :type: np.ndarray(ndim=2, dtype=float)[ell, freq]

    :maxlag: Maximum number of indices for which cross-correlation is computed
    :type: int

    Returns
    -------
    """
    # print(f"max frequency lag = {maxlag*dfreq:.2f} muHz")
    
    # corr_mat stores the correlation matrix [ell, time_chunk, lag]
    # corr_mat_gauss stores the gaussian fit [ell, time_chunk, lag]
    # corr_matarg stores the index corresponding to maximum corr [ell, time_chunk]
    # corr_matarg_gauss max corr for the gaussian fit [ell, time_chunk]
    
    corr_mat = np.zeros((4, 2*maxlag+1))
    corr_mat_gauss = np.zeros((4, 2*maxlag+1))
    corr_matarg = np.zeros(4)
    corr_matarg_gauss = np.zeros(4)
    
    lags_list, corr_list, corrnlist, corrbg_list = [], [], [], []
    for jdx in range(pfilt_list.shape[0]):
        p0 = [1., 0., 1., 0.]
        lags, corr = compute_cc(pfilt_list[jdx]/1e10, refarr/1e10, maxlag=maxlag)
        lags_bg, corr_bg = compute_cc(pexcl_list[jdx]/1e10, pfilt_list[jdx]/1e10, maxlag=maxlag)
        corr = corr + corr_bg
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
    """Gaussian profile given mean and fwhm

    Parameters
    ----------
    :x: range over which gaussian is computed
    :type: np.ndarray(ndim=1, dtype=float)

    :mu: location of gaussian peak
    :type: float

    :fwhm: FWHM of gaussian
    :type: float

    Returns
    -------
    gaussian profile
    """
    sigma = fwhm / np.sqrt(8. * np.log(2.))
    return np.exp(-(x-mu)**2/2./sigma/sigma)


def get_pslbg(SPS, visibility_matrix=True, return_nl_list=True):
    """Get the components of power spectrum. The power-spectrum is modelled as a linear combination
    of the following components.
    (1) Lorentzians for the modes
    (2) Harvey-like profiles for the background
    (3) Photon-noise

    This function returns a list psl_bg, where each element of the list corresponds to the
    power-spectrum of the above 3 components.

    Parameters
    ----------
    :SPS: instance of class src.model.cross_covariance.solarPS
    :type: solarPS

    :visibility_matrix: Set True to include corrections due to visibility matrix
    :type: bool (default: True)

    :return_nl_list: flag for returning list of enn and ell
    :type: bool (default: True)

    Returns
    -------
    :psl_bg: Each element of the list is either a Lorentzian for a specific (n, ell, m)
             combination or either of the background terms (2, 3) mentioned above.
    :type: list[np.ndarray(ndim=1)]

    :fmhz: The frequency bins for which the power spectrum is constructed.
    :type: np.ndarray(ndim=1)
    :unit: mHz
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
    """
    Noisify the input array with a chi2-2dof distribution

    Parameters
    ----------
    :iparr: Input spectra
    :type: np.ndarray(ndim=1, dtype=float)

    Returns
    -------
    noisy_arr
    :noisy_arr: noisy counterpart of input spectra
    :type: np.ndarray(ndim=1, dtype=float)
    """
    noise = np.random.randn(*(iparr.shape))**2
    noisy_arr = iparr*noise
    return noisy_arr


def gaussian(x, *p):
    """Creates a gaussian with the defined parameters. Useful for 
    passing the function to scipy.optimize.curve_fit

    Parameters
    ----------
    :x: domain on which gaussian is defined
    :type: np.ndarray(ndim=1, dtype=float)

    :p: parameters corresponding to the gaussian
    :type: list(len=4)
        p[0] = Amplitude of gaussian
        p[1] = centroid location
        p[2] = sigma
        p[3] = dc shift

    Returns
    -------
    gaussian profile on x
    """
    A, mu, sigma, k = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + k


def compute_cc(arr1, arr2, maxlag=20):
    """Computes the cross-correlation for lags in the range (-maxlag, maxlag+1)

    Parameters
    ----------
    :arr1: the raw power spectrum
    :type: np.ndarray(ndim=1, dtype=float)

    :arr2: the filter power spectrum model
    :type: np.ndarray(ndim=1, dtype=float)

    :maxlag: the maximum lag index
    :type: int
    :default: 20

    Returns
    -------
    lags, cc

    :lags: array containing list of lags
    :type: np.ndarray(ndim=1, dtype=int)
    :note: lags = np.arange(-maxlag, maxlag+1)

    :cc: cross-correlation array
    :type: np.ndarray(ndim=1, dtype=float)
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


if __name__ == "__main__":

    ## TODO LIST
    # [x]  amplitudes and lorentzians for different ell
    # [x]  amplitudes and lorentizians for different (ell, emm) combos
    # [x]  load gfilter
    # [x]  separate background fitting
    # [ ]  add frequency filtering -- better to add a butterworth filter,
    #      rather than just filtering out a part of the spectrum
    #      -- cross-correlation should be fine as long as 0-padding is done
    # [ ]  integrate fitting using peak-bagged frequencies
    # [ ] realistic estimation of errors in delnu

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
    bsp = np.load('/scratch/seismo/kashyap/processed/sun-intg/bsp-basis/bsp_knotnum_15.npy')
    amps_llk[-2:] = bgamps
    amps_llk = amps*1.0

    SPS = solarPS(lmax=ARGS.lmax,
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
    shiftval_list = [5, 10, 15, 20]
    shiftval = -shiftval_list[0]
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
    pexcl_list = [pexc0, pexc1, pexc2, pexc3]
    pfilt_list = np.array(pfilt_list)
    pexcl_list = np.array(pexcl_list)

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


    collect_fitval = []
    for idx in tqdm(range(ARGS.realizations), desc='MonteCarlo combined'):
        __a0, __a00 = get_freqlags(noisify(pmod0123s), np.array([psfit,]))
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

    axs[1].hist((collect0fitval_corr + shiftval)*dfreq, histtype=u'step', label='ell=0', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].hist((collect1fitval_corr + shiftval)*dfreq, histtype=u'step', label='ell=1', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].hist((collect2fitval_corr + shiftval)*dfreq, histtype=u'step', label='ell=2', bins=np.linspace(-2, 2, 100), density=True)
    axs[1].hist((collect3fitval_corr + shiftval)*dfreq, histtype=u'step', label='ell=3', bins=np.linspace(-2, 2, 100), density=True)
    fig.supxlabel('$\\delta\\nu^\\mathrm{pred} - \\delta\\nu^\\mathrm{true}$ in $\\mu$Hz', fontsize=12)
    for _axs in axs: _axs.legend()
    fig.tight_layout()
