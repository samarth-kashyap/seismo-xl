import sys
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils import read_a2z
from src.utils import read_bgparams
from scipy.optimize import curve_fit

# Local imports
from src.globalvars import globalVars
from src.stellarspec import stellarPS

# Defining some global variables
GVARS = globalVars()
# mode data is written as enn, ell, freq, A, gamma

#--------------------- ARGUMENT PARSER ---------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Navg', type=int, default=90 ,help='Length of sub-series (days)')
parser.add_argument('--Nshift', type=int, default=15, help='Shift between sub-series (days)')
parser.add_argument('--inclang', type=float, default=45., help='Inclination angle')
parser.add_argument('--freqmin', type=float, default=0.5, help='Minimum freq in mHz')
parser.add_argument('--freqmax', type=float, default=5.5, help='Maximum freq in mHz')
ARGS = parser.parse_args()
#----------------------------------------------------------------------------
assert ARGS.freqmin>0. and ARGS.freqmax<10., "Min freq out of range"
assert ARGS.freqmax>0. and ARGS.freqmax<10., "Max freq out of range"
assert ARGS.freqmax>ARGS.freqmin, "maxfreq < minfreq; exiting"

scratch_dir = f"/scratch/seismo/kashyap/processed/sun-intg"

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
    :SPS: instance of class src.stellarspec.stellarPS
    :type: stellarPS

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
    bgtype = 'apollinaire'
    bgl.append(np.loadtxt(f'./results/{suffix}/background.dat'))
    print(bgl[-1].shape)
    #kwargs = {'n_harvey': 2,
    #          'param': read_bgparams(f'./results/{suffix}/mcmc_background_mean.dat')}
    #bgl.append(SPS.get_background_lowfreq(A1=1.0, A2=0.0, Ap=0.0, type=bgtype, **kwargs))
    #bgl.append(SPS.get_background_lowfreq(A1=0.0, A2=1.0, Ap=0.0, type=bgtype, **kwargs))

    for _ell in range(SPS.lmax+1):
        psl, enns, ells, nus, gammas = SPS.construct_ps_list(ell=_ell,
                                                             visibility_matrix=visibility_matrix,
                                                             return_nl_list=return_nl_list)
        psl_ells = [*psl_ells, *psl[0]]
        psl_nlm = [*psl_nlm, *psl[1]]
        ell_list = [*ell_list, *ells]
        enn_list = [*enn_list, *enns]
        nu_list = [*nu_list, *nus]
        gamma_list = [*gamma_list, *gammas]

        for _enn in enns:
            _psnlm = SPS.construct_ps_normed_nlm(enn=_enn, ell=_ell, shiftfreq=0.,
                                                 scalefwhm=1., stahn=True)
            ps_nlm_dict[f"{_ell:d}-{_enn:02d}"] = _psnlm
            

    psl_bg = [*psl_ells, *bgl]
    print(psl_bg[0].shape)
    psl_nlm = [*psl_nlm, *bgl]
    ps_nlm_dict["bg1"] = bgl[0]
    #ps_nlm_dict["bg2"] = bgl[1]
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


def compute_delnu(pmod_list, pfilt_list, pexcl_list, corrected=False):
    # Computing delnu with correction factor
    collect0fitval = []
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
    suffix = "longts"
    suffix = f"N{int(ARGS.Navg)}-s{int(ARGS.Nshift)}"
    mode_dict, mode_cols = read_a2z(f'/data/seismo/kashyap/codes/p11-seismo-xl/results/{suffix}/modes_param.a2z')

    enn = mode_dict[:, 0].astype('int')
    ell = mode_dict[:, 1].astype('int')
    nus  = mode_dict[:, 2]*1e-6
    amps = mode_dict[:, 3]
    fwhm  = mode_dict[:, 4]*1e-6

    day2sec = 24*3600.
    rot_period = 31.71*day2sec # days (A&A 682, A67, 2024 - Breton, Lanza, Messina)
    a1rot = 1/rot_period
    pschunks = np.load(f'data/pschunks-8006161-{suffix}.npy')*1e-9


    # ?? Temporary fix; this is not needed in the next version
    if suffix=='longts':
        pschunks *= 1e-9
    fref = np.load(f'data/freq-8006161-{suffix}.npy')
    cond  = (fref*1e6>150.)*(fref*1e6<6000.)
    fref = fref[cond]
    pschunks = pschunks[:, cond]
    pref = pschunks.mean(axis=0)

    SPS = stellarPS(fref, 
                    lmax=3,
                    mode_ell=ell*1,
                    mode_enn=enn*1,
                    mode_nu=nus*1.,
                    mode_fwhm=fwhm*1.,
                    mode_sigfwhm=fwhm*0.1,
                    incl_angle=ARGS.inclang*np.pi/180.,
                    a1rot=a1rot,
                    a3rot=0.,
                    mode_max_nu=5.e-3,)
    freq_arr = SPS.nu_plus
    freq_mhz = freq_arr*1e3
    MASK_FREQ = (freq_mhz>=ARGS.freqmin)*(freq_mhz<=ARGS.freqmax)
    freq_arr = freq_arr[MASK_FREQ]
    freq_mhz = freq_mhz[MASK_FREQ]

    gfilter = gaussian_gfilt(freq_arr*1e3, 3.616, 1.5)
    print(f'LOADING SUCESS')
    psdict, fmhz, enn_list, ell_list, nu_list, gamma_list = get_pslbg(SPS, return_nl_list=True)
    psl_bg, ps_nlm_dict = psdict
    psl_bg = psl_bg[:, MASK_FREQ]
    pschunks = pschunks[:, MASK_FREQ]*gfilter[None, :]

    psfit = (amps @ psl_bg[:-1] + psl_bg[-1])*gfilter
    pref_arr = pref[MASK_FREQ]*gfilter

    assert np.prod(ell_list==ell), "Loaded amplitudes for ell dont match the current ell_list"
    assert np.prod(enn_list==enn), "Loaded amplitudes for enn dont match the current enn_list"

    mask0 = np.array(ell_list)==0
    mask1 = np.array(ell_list)==1
    mask2 = np.array(ell_list)==2
    mask3 = np.array(ell_list)==3

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(freq_mhz, pschunks[0]/gfilter, 'k', label='Observed')
    axs[0].plot(freq_mhz, psfit/gfilter, 'r', label='Model')
    axs[0].legend()
    axs[0].set_title('Power spectrum')
    axs[1].plot(freq_mhz, pschunks.mean(axis=0)/psfit, 'k')
    axs[1].set_title('$P^\\mathrm{model}/P^\\mathrm{observed}$')
    for _axs in axs:
        _axs.set_xscale('log')
        _axs.set_yscale('log')
    axs[1].set_xticks(np.arange(1, 5))
    fig.supxlabel('Frequency [mHz]')
    fig.supylabel('Power [ppm^2/muHz]')

    bgfit = psl_bg[-1]*1.
    print(f"----Number of frequency bins = {len(freq_arr)}")

    # Computing the MI filters
    pfilt0 = np.squeeze((amps[mask0] @ psl_bg[:-1][mask0, :] + bgfit)*gfilter)
    pfilt1 = np.squeeze((amps[mask1] @ psl_bg[:-1][mask1, :] + bgfit)*gfilter)
    pfilt2 = np.squeeze((amps[mask2] @ psl_bg[:-1][mask2, :] + bgfit)*gfilter)
    pfilt3 = np.squeeze((amps[mask3] @ psl_bg[:-1][mask3, :] + bgfit)*gfilter)
    pfilt_list = [pfilt0, pfilt1, pfilt2, pfilt3]
    pfilt_list = np.array(pfilt_list)

    # Computing the LC term
    pexc0 = np.squeeze((psfit - amps[mask0] @ psl_bg[:-1][mask0, :] - bgfit)*gfilter)
    pexc1 = np.squeeze((psfit - amps[mask1] @ psl_bg[:-1][mask1, :] - bgfit)*gfilter)
    pexc2 = np.squeeze((psfit - amps[mask2] @ psl_bg[:-1][mask2, :] - bgfit)*gfilter)
    pexc3 = np.squeeze((psfit - amps[mask3] @ psl_bg[:-1][mask3, :] - bgfit)*gfilter)
    pexcl_list = [pexc0, pexc1, pexc2, pexc3]
    pexcl_list = np.array(pexcl_list)
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
        __a0, __a00 = get_freqlags(noisify(pfilt0123s), np.array([psfit*gfilter,]))
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
