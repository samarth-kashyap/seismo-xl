import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Local imports
from sgkutils import readh5, saveh5
from src.utils import read_a2z
from src.stellarspec import stellarPS

#--------------------- ARGUMENT PARSER ---------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kic', type=int, default=8006161, help='Kepler KIC')
parser.add_argument('--lmax', type=int, default=3, help='Max ell observed in data')
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

kicstr = f"{ARGS.kic:09d}"
PAPDIR = "/scratch/seismo/kashyap/cloud/Yandex.Disk/papers-posters-docs/2025-seismo-xl"
scratch_dir = f"/scratch/seismo/kashyap/processed/p11-seismo-xl/{kicstr}"
try:
    santos_data = pd.read_csv(f'./data/santos2018-{kicstr}.csv')
except FileNotFoundError:
    pass


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
        lags, corr = compute_cc(pfilt_list[jdx], refarr, maxlag=maxlag)
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
    bgl.append(np.loadtxt(f'{scratch_dir}/peakbag-{suffix}/background.dat'))
    print(bgl[-1].shape)

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


def get_freqlags_corrected(refarr, pfilt_list, pexcl_list, maxlag=20):
    # print(f"max frequency lag = {maxlag*dfreq:.2f} muHz")
    
    # corr_mat stores the correlation matrix [ell, time_chunk, lag]
    # corr_mat_gauss stores the gaussian fit [ell, time_chunk, lag]
    # corr_matarg stores the index corresponding to maximum corr [ell, time_chunk]
    # corr_matarg_gauss max corr for the gaussian fit [ell, time_chunk]
    
    corr_mat = np.zeros((ARGS.lmax, 2*maxlag+1))
    corr_mat_gauss = np.zeros((ARGS.lmax, 2*maxlag+1))
    corr_matarg = np.zeros(ARGS.lmax)
    corr_matarg_gauss = np.zeros(ARGS.lmax)
    
    corrbg_list = []
    for jdx in range(ARGS.lmax):
        p0 = [1., 0., 1., 0.]
        coeff = [0., 0., 0., 0.]
        lags, corr = compute_cc(pfilt_list[jdx], refarr, maxlag=maxlag)
        lags_bg, corr_bg = compute_cc(pexcl_list[jdx], pfilt_list[jdx], maxlag=maxlag)
        corr = corr - corr_bg
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


def compute_delnu(pschunks, pfilt_list, pexcl_list, dfreq, maxlag=20):
    corr_mat = np.zeros((ARGS.lmax, pschunks.shape[0], 2*maxlag+1))
    corr_mat_gauss = np.zeros((ARGS.lmax, pschunks.shape[0], 2*maxlag+1))
    corr_matarg = np.zeros((ARGS.lmax, pschunks.shape[0]))
    corr_matarg_gauss = np.zeros((ARGS.lmax, pschunks.shape[0]))

    for idx in range(pschunks.shape[0]):
        (_cm, _cmg), (_cma, _cmag), _cbg_list = get_freqlags_corrected(pschunks[idx], pfilt_list, pexcl_list, maxlag=maxlag)
        corr_mat[:, idx, :] = _cm
        corr_mat_gauss[:, idx, :] = _cmg
        corr_matarg[:, idx] = _cma
        corr_matarg_gauss[:, idx] = _cmag
    domega_muhz = -corr_matarg_gauss*dfreq/muhz2hz
    return domega_muhz



def compute_errors_montecarlo(psfit ,pfilt_list, pexcl_list, dfreq, maxlag=20, samples=10000):
    dom_list = []
    for idx in tqdm(range(samples), desc='Computing errors using MonteCarlo'):
        _psc = noisify(psfit)
        _dom = compute_delnu(_psc[None, :], pfilt_list, pexcl_list, dfreq, maxlag=maxlag)
        dom_list.append(_dom)
    dom_list = np.squeeze(np.array(dom_list))
    dom_list[np.isnan(dom_list)] = 0.
    domega_sig = np.zeros(ARGS.lmax)
    for idx in range(ARGS.lmax):
        mask = abs(dom_list[:, idx])<5.
        assert (~mask).sum()/mask.sum() < 0.01, "More than 1% outliers"
        domega_sig[idx] = np.std(dom_list[mask, idx], axis=0)
    return dom_list, domega_sig


def moving_avg(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w


if __name__ == "__main__":
    kicstr = f"{ARGS.kic:09d}"
    suffix = f"N{int(ARGS.Navg)}-s{int(ARGS.Nshift)}"
    mode_dict, mode_cols = read_a2z(f'{scratch_dir}/peakbag-{suffix}/modes_param.a2z')
    obsdata = readh5(f'{scratch_dir}/kplr{kicstr}-{suffix}.h5')

    muhz2hz = 1e-6
    nhz2hz = 1e-9
    mhz2hz = 1e-3
    day2sec = 24*3600.

    enn = mode_dict[:, 0].astype('int')
    ell = mode_dict[:, 1].astype('int')
    nus  = mode_dict[:, 2]*muhz2hz
    amps = mode_dict[:, 3]
    fwhm  = mode_dict[:, 4]*muhz2hz

    rot_period = 31.71*day2sec # days (A&A 682, A67, 2024 - Breton, Lanza, Messina)
    a1rot = 1/rot_period
    pschunks = obsdata['pschunks']*nhz2hz
    fref = obsdata['freq']
    cond = obsdata['fmask']
    fref = fref[cond]
    pschunks = pschunks[:, cond]
    pref = pschunks.mean(axis=0)

    SPS = stellarPS(fref, 
                    lmax=ARGS.lmax,
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

    numax = np.loadtxt(f'{scratch_dir}/peakbag-{suffix}/background_parameters.dat')[-3][0]*muhz2hz
    gfilter = gaussian_gfilt(freq_arr/mhz2hz, numax/mhz2hz, 1.5)
    print(f'LOADING SUCESS')
    psdict, fmhz, enn_list, ell_list, nu_list, gamma_list = get_pslbg(SPS, return_nl_list=True)
    psl_bg, ps_nlm_dict = psdict
    psl_bg = psl_bg[:, MASK_FREQ]
    pschunks = pschunks[:, MASK_FREQ]*gfilter[None, :]

    psfit = (amps @ psl_bg[:-1] + psl_bg[-1])*gfilter
    pref_arr = pref[MASK_FREQ]*gfilter
    bgfit = psl_bg[-1]*1.
    print(f"----Number of frequency bins = {len(freq_arr)}")
    assert np.prod(ell_list==ell), "Loaded amplitudes for ell dont match the current ell_list"
    assert np.prod(enn_list==enn), "Loaded amplitudes for enn dont match the current enn_list"

    maskell_list = []
    pfilt_list = []
    pexcl_list = []
    for idx in range(ARGS.lmax):
        _mask = np.array(ell_list)==idx
        maskell_list.append(_mask)
        pfilt_list.append(np.squeeze((amps[_mask] @ psl_bg[:-1][_mask, :] + bgfit)*gfilter))
        pexcl_list.append(np.squeeze((psfit - amps[_mask] @ psl_bg[:-1][_mask, :] - bgfit)*gfilter))

    pfilt_list = np.array(pfilt_list)
    pexcl_list = np.array(pexcl_list)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].plot(freq_mhz, pschunks[0]*1e9, 'k', label='Observed')
    axs[0].plot(freq_mhz, psfit*1e9, 'r', label='Model')
    axs[0].legend()
    axs[0].set_title('Power spectrum')
    axs[1].plot(freq_mhz, pschunks.mean(axis=0)/psfit, 'k')
    axs[1].set_title('$P^\\mathrm{observed}/P^\\mathrm{model}$')
    for _axs in axs:
        _axs.set_xscale('linear')
        _axs.set_yscale('linear')
    axs[1].set_xticks(np.arange(1, 5))
    fig.supxlabel('Frequency [mHz]')
    fig.supylabel('Power [ppm^2/muHz]')

    maxlag = 26
    dfreq = freq_arr[1] - freq_arr[0]
    domega_muhz = compute_delnu(pschunks, pfilt_list, pexcl_list, dfreq, maxlag=maxlag)
    domega_mc_list, domega_sig = compute_errors_montecarlo(psfit, pfilt_list, pexcl_list, dfreq, maxlag=maxlag)

    opdict = {}
    opdict['domega_muhz'] = domega_muhz
    opdict['domega_sig'] = domega_sig
    opdict['domega_mc_list'] = domega_mc_list
    opdict['time_arr'] = obsdata['tmid_list']
    saveh5(f"{scratch_dir}/delnu-{kicstr}-{suffix}.h5", opdict)

    santos_data_avg = {}
    for key in santos_data.keys():
        santos_data_avg[key] = moving_avg(santos_data[key], int(ARGS.Navg//90))

    time_arr = obsdata['tmid_list']
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharex=True, sharey=True)
    for idx in range(ARGS.lmax):
        fint = interp1d(santos_data_avg['time'], santos_data_avg[f'delnu{idx}'], bounds_error=False,)
        finte = interp1d(santos_data_avg['time'], santos_data_avg[f'delnu{idx}_error'], bounds_error=False)
        pbobs = fint(time_arr)
        pbobse = finte(time_arr)
        
        #axs[idx].plot(time_arr, domega_muhz[idx], 'o-k')
        axs[idx].set_title('$\\ell$ = ' + f'{idx}')
        axs[idx].errorbar(time_arr, pbobs, yerr=pbobse, capsize=5, color='r', marker='o', markersize=4, linestyle='') 
        axs[idx].errorbar(time_arr, domega_muhz[idx], yerr=np.ones_like(domega_muhz[idx])*domega_sig[idx], capsize=5, color='k', marker='o', markersize=4, linestyle='') 
        axs[idx].plot(time_arr, domega_muhz[idx], 'o-k')
    fig.supxlabel('Time [day]', fontsize=14)
    fig.supylabel('$\\delta\\omega_\\ell$ in $\\mu$Hz', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{PAPDIR}/delnu-comparison-{kicstr}.png')