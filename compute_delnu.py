import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize

# Local imports
import h5py


def readh5(fname):
    """Load an HDF5 file into a nested Python dictionary.

    Parameters
    ----------
    fname : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary mirroring the HDF5 group/dataset hierarchy.
        Scalar datasets are returned as Python scalars; byte strings are
        decoded to ``str``.
    """
    def _load(group, path='/'):
        result = {}
        for key in group[path].keys():
            fp = f"{path}/{key}" if path != '/' else f"/{key}"
            item = group[fp]
            if isinstance(item, h5py.Group):
                result[key] = _load(group, fp)
            else:
                data = item[()]
                if isinstance(data, bytes):
                    result[key] = data.decode('utf-8')
                elif isinstance(data, np.ndarray) and data.size == 1:
                    result[key] = data.item()
                elif isinstance(data, np.ndarray) and len(data.shape) == 1:
                    result[key] = data.tolist() if data.dtype.kind in 'biufc' else data
                else:
                    result[key] = data
        return result
    with h5py.File(fname, 'r') as f:
        return _load(f)


def saveh5(fname, dictionary):
    """Save a nested Python dictionary to an HDF5 file.

    Parameters
    ----------
    fname : str
        Destination file path.
    dictionary : dict
        Arbitrarily nested dictionary. Supported leaf types: ``np.ndarray``,
        ``list``, ``tuple``, ``str``, ``bytes``, ``int``, ``float``, ``bool``,
        and ``np.number`` subclasses.
    """
    def _save(d, group, path='/'):
        for key, val in d.items():
            fp = f"{path}/{key}" if path != '/' else f"/{key}"
            if isinstance(val, dict):
                _save(val, group, fp)
            elif isinstance(val, (np.ndarray, list, tuple)):
                group.create_dataset(fp, data=np.array(val))
            elif isinstance(val, (str, bytes)):
                group.create_dataset(fp, data=np.bytes_(val))
            elif isinstance(val, (int, float, bool, np.number)):
                group.create_dataset(fp, data=val)
            else:
                raise TypeError(f"Unsupported type {type(val)} for key {key}")
    with h5py.File(fname, 'w') as f:
        _save(dictionary, f)
from src.utils import read_a2z
from src.stellarspec import stellarPS
from src.config import Config

PARAMS = Config('./config.yml')


def get_freqlags(refarr, pfilt_list, maxlag=20):
    """Compute frequency lags via cross-correlation and Gaussian fitting.

    Parameters
    ----------
    refarr : np.ndarray, shape (N,)
        Reference (observed) power spectrum.
    pfilt_list : np.ndarray, shape (lmax, N)
        Matched-filter power spectra for each harmonic degree.
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.

    Returns
    -------
    corr_mats : tuple of np.ndarray
        ``(corr_mat, corr_mat_gauss)`` - raw and Gaussian-fitted correlation
        matrices of shape ``(lmax, 2*maxlag+1)``.
    corr_args : tuple of np.ndarray
        ``(corr_matarg, corr_matarg_gauss)`` - lag indices of peak correlation
        and peak of the Gaussian fit, each of shape ``(lmax,)``.
    """
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
    """Evaluate a Gaussian profile given a mean and FWHM.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Domain over which the Gaussian is evaluated.
    mu : float
        Location of the Gaussian peak.
    fwhm : float
        Full width at half maximum of the Gaussian (same units as ``x``).

    Returns
    -------
    np.ndarray, shape (N,)
        Gaussian profile with unit peak amplitude.
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
            _psnlm = SPS.construct_ps_normed_nlm(enn=_enn, ell=_ell, shiftfreq=0., scalefwhm=1.,)
            ps_nlm_dict[f"{_ell:d}-{_enn:02d}"] = _psnlm
            

    psl_bg = [*psl_ells, *bgl]
    print(psl_bg[0].shape)
    psl_nlm = [*psl_nlm, *bgl]
    ps_nlm_dict["bg1"] = bgl[0]
    psl_bg = np.array(psl_bg)
    if return_nl_list:
        return (psl_bg, ps_nlm_dict), fmhz, enn_list, ell_list, nu_list, gamma_list
    else:
        return psl_bg, fmhz


def noisify(iparr):
    """Apply chi-squared (2 d.o.f.) noise to a power spectrum.

    Each frequency bin is multiplied by an independent draw from a
    chi-squared distribution with 2 degrees of freedom (i.e. the square
    of a standard-normal variate), as appropriate for an exponentially
    distributed power spectrum.

    Parameters
    ----------
    iparr : np.ndarray
        Input (model) power spectrum.

    Returns
    -------
    noisy_arr : np.ndarray
        Noise-realization of the input spectrum, same shape as ``iparr``.
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


def lorentzian(x, *p):
    """Evaluate a Lorentzian (Cauchy) profile with an additive DC offset.

    Convenience wrapper for use with ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    x : np.ndarray
        Domain on which the Lorentzian is evaluated.
    *p : float
        Four parameters ``(A, mu, sigma, k)`` where

        * ``A``     - amplitude,
        * ``mu``    - centroid,
        * ``sigma`` - half-width at half maximum,
        * ``k``     - DC offset.

    Returns
    -------
    np.ndarray
        Lorentzian profile on ``x``.
    """
    A, mu, sigma, k = p
    return A/(1 + ((x-mu)/sigma)**2) + k


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


def get_freqlags_corrected(refarr, pfilt_list, pexcl_list, fitfunc, maxlag=20):
    """Compute background-corrected frequency lags via cross-correlation.

    The cross-correlation between the filter and the excluded-mode spectrum is
    subtracted as a background correction before fitting.

    Parameters
    ----------
    refarr : np.ndarray, shape (N,)
        Reference (observed) power spectrum.
    pfilt_list : np.ndarray, shape (lmax, N)
        Matched-filter power spectra for each harmonic degree.
    pexcl_list : np.ndarray, shape (lmax, N)
        Leakage-correction (excluded-mode) spectra for each harmonic degree.
    fitfunc : callable
        Profile function used to fit the cross-correlation peak
        (e.g. ``gaussian`` or ``lorentzian``).
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.

    Returns
    -------
    corr_mats : tuple of np.ndarray
        ``(corr_mat, corr_mat_gauss)`` - raw corrected and fitted correlation
        matrices of shape ``(lmax, 2*maxlag+1)``.
    corr_args : tuple of np.ndarray
        ``(corr_matarg, corr_matarg_gauss)`` - peak-lag indices of shape
        ``(lmax,)`` from argmax and from the fitted profile.
    corrbg_list : list of np.ndarray
        Background cross-correlation array for each harmonic degree.
    """
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
            coeff, var_matrix = curve_fit(fitfunc, lags, corr, p0=p0)
        except RuntimeError:
            coeff[1] = np.nan
        corr_mat_gauss[jdx, :] = gaussian(lags, *coeff)
        corr_mat[jdx, :] = corr
        max_idx = np.argmax(corr)
        corr_matarg[jdx] = lags[max_idx]
        corr_matarg_gauss[jdx] = coeff[1]
        corrbg_list.append(corr_bg)
    return (corr_mat, corr_mat_gauss), (corr_matarg, corr_matarg_gauss), corrbg_list



def get_freqlags_polyfit(refarr, pfilt_list, pexcl_list, maxlag=20):
    """Compute background-corrected frequency lags using polynomial peak fitting.

    Fits a degree-5 polynomial to the normalised, background-corrected
    cross-correlation and locates the peak via ``scipy.optimize.minimize``.

    Parameters
    ----------
    refarr : np.ndarray, shape (N,)
        Reference (observed) power spectrum.
    pfilt_list : np.ndarray, shape (lmax, N)
        Matched-filter power spectra for each harmonic degree.
    pexcl_list : np.ndarray, shape (lmax, N)
        Leakage-correction (excluded-mode) spectra for each harmonic degree.
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.

    Returns
    -------
    corr_mats : tuple of np.ndarray
        ``(corr_mat, corr_mat_pf)`` - normalised corrected cross-correlation
        and polynomial-fitted cross-correlation, each of shape
        ``(lmax, 2*maxlag+1)``.
    corr_args : tuple of np.ndarray
        ``(corr_matarg, corr_matarg_pf)`` - argmax and polynomial-fit peak
        positions, each of shape ``(lmax,)``.
    corrbg_list : list of np.ndarray
        Background cross-correlation array for each harmonic degree.
    """
    corr_mat_pf = np.zeros((ARGS.lmax, 2*maxlag+1))
    corr_matarg = np.zeros(ARGS.lmax)
    corr_matarg_pf = np.zeros(ARGS.lmax)
    
    corrbg_list = []
    for jdx in range(ARGS.lmax):
        coeff = [0., 0., 0., 0.]
        lags, corr = compute_cc(pfilt_list[jdx], refarr, maxlag=maxlag)
        lags_bg, corr_bg = compute_cc(pexcl_list[jdx], pfilt_list[jdx], maxlag=maxlag)
        corr = corr + corr_bg
        try:
            _corr = corr - corr.min()
            _corr = _corr/_corr.max()
            pf = np.polyfit(lags, _corr, deg=5)
            mfunc = lambda x: -1*np.polyval(pf, x)
            fit = minimize(mfunc, x0=0)
        except RuntimeError:
            coeff[1] = np.nan
        corr_mat[jdx, :] = _corr
        corr_mat_pf[jdx, :] = -1*mfunc(lags)
        max_idx = np.argmax(corr)
        corr_matarg[jdx] = lags[max_idx]
        corr_matarg_pf[jdx] = fit.x[0]
        corrbg_list.append(corr_bg)
    return (corr_mat, corr_mat_pf), (corr_matarg, corr_matarg_pf), corrbg_list


def compute_delnu(pschunks, pfilt_list, pexcl_list, dfreq, maxlag=20, fittype='gaussian'):
    """Compute frequency shifts (delta-nu) for each harmonic degree and time chunk.

    For every time chunk in ``pschunks`` the background-corrected
    cross-correlation is computed and the peak is located with the requested
    fitting method, yielding a frequency shift in microHz.

    Parameters
    ----------
    pschunks : np.ndarray, shape (T, N)
        Observed power-spectrum chunks, one per time interval.
    pfilt_list : np.ndarray, shape (lmax, N)
        Matched-filter power spectra for each harmonic degree.
    pexcl_list : np.ndarray, shape (lmax, N)
        Leakage-correction spectra for each harmonic degree.
    dfreq : float
        Frequency resolution in Hz (used to convert lag indices to microHz).
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.
    fittype : {'gaussian', 'lorentzian', 'polynomial'}, optional
        Profile function used to locate the cross-correlation peak.

    Returns
    -------
    domega_muhz : np.ndarray, shape (lmax, T)
        Frequency shifts in microHz for each (harmonic degree, time chunk)
        combination.
    """
    corr_mat_gauss = np.zeros((ARGS.lmax, pschunks.shape[0], 2*maxlag+1))
    corr_matarg = np.zeros((ARGS.lmax, pschunks.shape[0]))
    corr_matarg_gauss = np.zeros((ARGS.lmax, pschunks.shape[0]))

    for idx in range(pschunks.shape[0]):
        if fittype=='gaussian':
            (_cm, _cmg), (_cma, _cmag), _cbg_list = get_freqlags_corrected(pschunks[idx], pfilt_list, pexcl_list, gaussian, maxlag=maxlag)
        elif fittype=='lorentzian':
            (_cm, _cmg), (_cma, _cmag), _cbg_list = get_freqlags_corrected(pschunks[idx], pfilt_list, pexcl_list, lorentzian, maxlag=maxlag)
        elif fittype=='polynomial':
            (_cm, _cmg), (_cma, _cmag), _cbg_list = get_freqlags_polyfit(pschunks[idx], pfilt_list, pexcl_list, maxlag=maxlag)
        corr_mat[:, idx, :] = _cm
        corr_mat_gauss[:, idx, :] = _cmg
        corr_matarg[:, idx] = _cma
        corr_matarg_gauss[:, idx] = _cmag
    domega_muhz = -corr_matarg_gauss*dfreq/muhz2hz
    return domega_muhz



def compute_errors_montecarlo(psfit, pfilt_list, pexcl_list, dfreq, maxlag=20, samples=10000):
    """Estimate frequency-shift uncertainties via Monte Carlo noise realisations.

    Generates ``samples`` noise realisations of ``psfit`` using
    :func:`noisify`, computes ``compute_delnu`` for each, and returns the
    standard deviation across realisations as the uncertainty estimate.
    Outliers with ``|delta_omega| >= 5 muHz`` are excluded before computing
    the standard deviation.

    Parameters
    ----------
    psfit : np.ndarray, shape (N,)
        Best-fit (noiseless) model power spectrum.
    pfilt_list : np.ndarray, shape (lmax, N)
        Matched-filter power spectra for each harmonic degree.
    pexcl_list : np.ndarray, shape (lmax, N)
        Leakage-correction spectra for each harmonic degree.
    dfreq : float
        Frequency resolution in Hz.
    maxlag : int, optional
        Maximum lag index for the cross-correlation window.
    samples : int, optional
        Number of Monte Carlo realisations.

    Returns
    -------
    dom_list : np.ndarray, shape (samples, lmax)
        Frequency shifts for each realisation and harmonic degree.
    domega_sig : np.ndarray, shape (lmax,)
        Estimated 1-sigma uncertainty on the frequency shift for each
        harmonic degree.
    """
    for idx in tqdm(range(samples), desc='Computing errors using MonteCarlo'):
        _psc = noisify(psfit)
        _dom = compute_delnu(_psc[None, :], pfilt_list, pexcl_list, dfreq, maxlag=maxlag)
        dom_list.append(_dom)
    dom_list = np.squeeze(np.array(dom_list))
    dom_list[np.isnan(dom_list)] = 0.
    domega_sig = np.zeros(ARGS.lmax)
    for idx in range(ARGS.lmax):
        mask = abs(dom_list[:, idx])<5.
        outlier_frac = (~mask).sum()/mask.sum()
        if outlier_frac > 0.01: print(f"More than 1% outliers: {outlier_frac:.2f}")
        domega_sig[idx] = np.std(dom_list[mask, idx], axis=0)
    return dom_list, domega_sig


def moving_avg(x, w):
    """Compute a simple (boxcar) moving average.

    Parameters
    ----------
    x : array_like, shape (N,)
        Input data.
    w : int
        Window width.

    Returns
    -------
    np.ndarray, shape (N - w + 1,)
        Moving-average of ``x`` with window ``w``.
    """
    return np.convolve(x, np.ones(w), 'valid')/w



def plot_compare(time_arr, dom_santos, derr_santos, dom, derr):
    """Plot frequency-shift time series comparing this work against Santos et al. (2018).

    Parameters
    ----------
    time_arr : np.ndarray, shape (T,)
        Observation mid-times in days.
    dom_santos : list of np.ndarray
        Frequency shifts from Santos et al. (2018) for each harmonic degree.
    derr_santos : list of np.ndarray
        Uncertainties on ``dom_santos``.
    dom : np.ndarray, shape (lmax, T)
        Frequency shifts from this work.
    derr : np.ndarray, shape (lmax,)
        Uncertainty estimates for ``dom``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes, one per harmonic degree.
    """
    def get_dcs(arr1, arr2):
        _arr1, _arr2 = arr1*1., arr2*1.
        masknan = np.isnan(arr1) + np.isnan(arr2)
        _arr1[masknan] = 0.
        _arr2[masknan] = 0.
        darr = _arr1 - _arr2
        darr2 = darr**2
        return np.sqrt(np.nanmean(darr2))*np.ones_like(_arr1)

    # For fixing reference; 
    dcs = [get_dcs(np.array(dom_santos[idx]), dom[idx]) for idx in range(len(dom_santos))]
        
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), sharex=True, sharey=True)
    for idx in range(len(dom_santos)):
        axs[idx].set_title('$\\ell$ = ' + f'{idx}')
        axs[idx].errorbar(time_arr, dom_santos[idx]+dcs[idx], yerr=derr_santos[idx], 
                          capsize=5, color='r', marker='o', markersize=4, alpha=0.8, 
                          linestyle='', label='Santos et. al. (2018)') 
        axs[idx].errorbar(time_arr, dom[idx], yerr=np.ones_like(dom[idx])*derr[idx], 
                          capsize=5, color='k', marker='x', markersize=4, alpha=0.8, 
                          linestyle='', label='This work') 
        axs[idx].legend()
    fig.supxlabel('Time [day]', fontsize=14)
    fig.supylabel('$\\delta\\omega_\\ell$ in $\\mu$Hz', fontsize=14)
    fig.tight_layout()
    return fig, axs


def plot_compare_vertical(time_arr, dom_santos, derr_santos, dom, derr):
    """Plot frequency-shift time series in a vertical layout.

    Same data as :func:`plot_compare` but with harmonic degrees stacked
    vertically rather than horizontally.

    Parameters
    ----------
    time_arr : np.ndarray, shape (T,)
        Observation mid-times in days.
    dom_santos : list of np.ndarray
        Frequency shifts from Santos et al. (2018) for each harmonic degree.
    derr_santos : list of np.ndarray
        Uncertainties on ``dom_santos``.
    dom : np.ndarray, shape (lmax, T)
        Frequency shifts from this work.
    derr : np.ndarray, shape (lmax,)
        Uncertainty estimates for ``dom``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes, one per harmonic degree.
    """
    def get_dcs(arr1, arr2):
        _arr1, _arr2 = arr1*1., arr2*1.
        masknan = np.isnan(arr1) + np.isnan(arr2)
        _arr1[masknan] = 0.
        _arr2[masknan] = 0.
        darr = _arr1 - _arr2
        darr2 = darr**2
        return np.sqrt(np.nanmean(darr2))*np.ones_like(_arr1)

    # For fixing reference; 
    dcs = [get_dcs(np.array(dom_santos[idx]), dom[idx]) for idx in range(len(dom_santos))]
        
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4, 5.5), sharey=True)
    for idx in range(len(dom_santos)):
        #axs[idx].set_title('$\\ell$ = ' + f'{idx}')
        axs[idx].errorbar(time_arr, dom_santos[idx]+dcs[idx], yerr=derr_santos[idx], 
                          capsize=5, color='r', marker='o', markersize=4, alpha=0.8, 
                          linestyle='', label='Santos et. al. (2018)') 
        axs[idx].errorbar(time_arr, dom[idx], yerr=np.ones_like(dom[idx])*derr[idx], 
                          capsize=5, color='k', marker='x', markersize=4, alpha=0.8, 
                          linestyle='', label='This work') 
        _axs = axs[idx].twinx()
        _axs.set_ylabel('$\\ell$ = ' + f'{idx}', rotation=0, labelpad=15, fontsize=12)
        _axs.set_yticks(ticks=np.arange(4)*0.5 - 0.5, labels=[])
        axs[idx].legend()
        axs[idx].set_xticks(np.arange(400, 1410, 200))
    fig.supxlabel('Time [day]', fontsize=12)
    fig.supylabel('$\\delta\\omega_\\ell$ [$\\mu$Hz]', fontsize=12)
    fig.tight_layout()
    return fig, axs


def plot_cc(psc, psf, pse, tarr, numfitpix=10, fittype='gaussian', dfreq=1.e-6):
    """Plot cross-correlation profiles for each time chunk and harmonic degree.

    Stacks background-corrected cross-correlation curves vertically in time,
    optionally overlaying polynomial or Gaussian fits to the peak.

    Parameters
    ----------
    psc : np.ndarray, shape (T, N)
        Observed power-spectrum chunks.
    psf : np.ndarray, shape (lmax, N)
        Matched-filter spectra for each harmonic degree.
    pse : np.ndarray, shape (lmax, N)
        Leakage-correction spectra for each harmonic degree.
    tarr : np.ndarray, shape (T,)
        Mid-times corresponding to each chunk (used as vertical offsets).
    numfitpix : int, optional
        Half-width (in lag bins) of the fitting window around the peak.
    fittype : {'gaussian', 'polynomial'}, optional
        Profile used to fit the cross-correlation peak.
    dfreq : float, optional
        Frequency resolution in Hz (used to convert lags to microHz on axis).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes, one per harmonic degree.
    """
    for jdx in range(3):
        for idx in range(len(psc)):
            shifty = dt*idx
            lags, corr = compute_cc(psf[jdx], psc[idx])
            lags_bg, corr_bg = compute_cc(pse[jdx], psf[jdx])
            corr2 = corr - corr_bg
            corr2 -= corr2.min()
            corr2 /= corr2.max()
            corr2 *= t0
            maxidx = np.argmax(corr2)
            pslice = slice(max(0, maxidx-numfitpix), min(len(lags), maxidx+numfitpix))
            assert len(lags[pslice])>0, f"zero size, maxidx = {maxidx}, lags len = {len(lags)}"
            axs[jdx].plot(lags[pslice]*dfreq*1e6, corr2[pslice]+shifty, '+k', alpha=0.5)
            if fittype=='polynomial':
                pf = np.polyfit(lags[pslice], corr2[pslice], deg=5)
                mfunc = lambda x: -1*np.polyval(pf, x)
                axs[jdx].plot(lags[pslice]*dfreq*1e6, -1*mfunc(lags[pslice])+shifty, 'r', alpha=0.5)
            elif fittype=='gaussian':
                p0 = [1., 0., 1., 0.]
                coeff, var_matrix = curve_fit(gaussian, lags[pslice], corr2[pslice], p0=p0)
                axs[jdx].plot(lags[pslice]*dfreq*1e6, gaussian(lags[pslice], *coeff)+shifty, 'r')
            #axs[jdx].text((lags[0]-25)*dfreq*1e6, shifty, f't ={tarr[idx]:7.1f}d')
        axs[jdx].set_title('$\\ell =$' + f'{jdx}')
        #axs[jdx].set_xlim([(lags[0]-26)*dfreq*1e6, lags[-1]*dfreq*1e6])
    fig.supxlabel('Frequency lag [$\\mu$Hz]')
    fig.tight_layout()
    return fig, axs


def plot2cc(psc, psf, pse, tarr, numfitpix=10, fittype='gaussian', dfreq=1.e-6):
    """Plot cross-correlation profiles with frequency lag on the vertical axis.

    Like :func:`plot_cc` but transposes the axes so that the frequency lag
    appears on the y-axis and time (correlation amplitude) on the x-axis,
    also marking the fitted peak position for each chunk.

    Parameters
    ----------
    psc : np.ndarray, shape (T, N)
        Observed power-spectrum chunks.
    psf : np.ndarray, shape (lmax, N)
        Matched-filter spectra for each harmonic degree.
    pse : np.ndarray, shape (lmax, N)
        Leakage-correction spectra for each harmonic degree.
    tarr : np.ndarray, shape (T,)
        Mid-times corresponding to each chunk (used as horizontal offsets).
    numfitpix : int, optional
        Half-width (in lag bins) of the fitting window around the peak.
    fittype : {'gaussian', 'polynomial'}, optional
        Profile used to fit the cross-correlation peak.
    dfreq : float, optional
        Frequency resolution in Hz (used to convert lags to microHz on axis).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes, one per harmonic degree.
    """
    for jdx in range(3):
        for idx in range(len(psc)):
            shifty = dt*idx
            lags, corr = compute_cc(psf[jdx], psc[idx])
            lags_bg, corr_bg = compute_cc(pse[jdx], psf[jdx])
            corr2 = corr - corr_bg
            corr2 -= corr2.min()
            corr2 /= corr2.max()
            corr2 *= t0
            maxidx = np.argmax(corr2)
            pslice = slice(max(0, maxidx-numfitpix), min(len(lags), maxidx+numfitpix))
            assert len(lags[pslice])>0, f"zero size, maxidx = {maxidx}, lags len = {len(lags)}"
            axs[jdx].plot(corr2[pslice]+shifty, -lags[pslice]*dfreq*1e6, '+k', alpha=0.3)
            if fittype=='polynomial':
                pf = np.polyfit(lags[pslice], corr2[pslice], deg=5)
                mfunc = lambda x: -1*np.polyval(pf, x)
                axs[jdx].plot(-1*mfunc(lags[pslice])+shifty, -lags[pslice]*dfreq*1e6, 'r', alpha=0.5)
                fit = minimize(mfunc, x0=0)
                peakval = fit.x*dfreq*1e6
            elif fittype=='gaussian':
                p0 = [1., 0., 1., 0.]
                coeff, var_matrix = curve_fit(gaussian, lags[pslice], corr2[pslice], p0=p0)
                mfunc = lambda x: gaussian(x, *coeff)
                axs[jdx].plot(gaussian(lags[pslice], *coeff)+shifty, -lags[pslice]*dfreq*1e6, 'r')
                fit = minimize(mfunc, x0=0)
                peakval = fit.x*dfreq*1e6
            axs[jdx].plot(tarr[idx], -peakval, 'xk', markersize=5,)
        axs[jdx].set_title('$\\ell =$' + f'{jdx}')
        axs[jdx].set_xticks(np.arange(400, 1300, 200))
    fig.supylabel('Frequency lag [$\\mu$Hz]')
    fig.supxlabel('Time [day]')
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    #--------------------- ARGUMENT PARSER ---------------------------------
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--kic', type=int, default=8006161, help='Kepler KIC')
    parser.add_argument('--lmax', type=int, default=3, help='Max ell observed in data')
    parser.add_argument('--Navg', type=int, default=180,help='Length of sub-series (days)')
    parser.add_argument('--Nshift', type=int, default=45, help='Shift between sub-series (days)')
    parser.add_argument('--inclang', type=float, default=45., help='Inclination angle')
    parser.add_argument('--freqmin', type=float, default=0.5, help='Minimum freq in mHz')
    parser.add_argument('--freqmax', type=float, default=5.5, help='Maximum freq in mHz')
    parser.add_argument('--output-dir', type=str, default=PARAMS.output_dir,
                        help='Output directory (default: config.yml output_dir)')
    parser.add_argument('--papers-dir', type=str, default=None,
                        help='Directory for paper figures (default: {output_dir}/papers)')
    ARGS = parser.parse_args()
    #----------------------------------------------------------------------------
    assert ARGS.freqmin>0. and ARGS.freqmax<10., "Min freq out of range"
    assert ARGS.freqmax>0. and ARGS.freqmax<10., "Max freq out of range"
    assert ARGS.freqmax>ARGS.freqmin, "maxfreq < minfreq; exiting"

    kicstr = f"{ARGS.kic:09d}"
    scratch_dir = f"{ARGS.output_dir}/{kicstr}"
    papers_dir = ARGS.papers_dir if ARGS.papers_dir else f"{ARGS.output_dir}/papers"
    try:
        santos_data = pd.read_csv(f'./data/santos2018b-{kicstr}.csv')
    except FileNotFoundError:
        pass

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
    fref = np.array(obsdata['freq'])
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

    maxlag = PARAMS.maxlag
    dfreq = freq_arr[1] - freq_arr[0]

    domega1muhz = compute_delnu(pschunks, pfilt_list, pexcl_list, dfreq, 
                                maxlag=maxlag, fittype='gaussian')
    domega2muhz = compute_delnu(pschunks, pfilt_list, pexcl_list, dfreq, 
                                maxlag=maxlag, fittype='polynomial')
    domega3muhz = compute_delnu(pschunks, pfilt_list, pexcl_list, dfreq, 
                                maxlag=maxlag, fittype='lorentzian')
    domega_mc_list, domega_sig = compute_errors_montecarlo(psfit, pfilt_list, 
                                                           pexcl_list, dfreq, 
                                                           maxlag=maxlag, 
                                                           samples=PARAMS.samples)

    opdict = {}
    opdict['domega_muhz_gaussian'] = domega1muhz
    opdict['domega_muhz_polynomial'] = domega2muhz
    opdict['domega_muhz_lorentzian'] = domega3muhz
    opdict['domega_sig'] = domega_sig
    opdict['domega_mc_list'] = domega_mc_list
    opdict['time_arr'] = obsdata['tmid_list']
    opdict['inclination'] = ARGS.inclang
    saveh5(f"{scratch_dir}/delnu-{kicstr}-{suffix}.h5", opdict)

    santos_data_avg = {}
    for key in santos_data.keys():
        santos_data_avg[key] = moving_avg(santos_data[key], max(1, int(ARGS.Navg//90)))

    time_arr = obsdata['tmid_list']
    domega_muhz_santos = []
    domega_err_santos = []
    for idx in range(ARGS.lmax):
        fint = interp1d(santos_data_avg['time'], santos_data_avg[f'delnu{idx}'], bounds_error=False,)
        finte = interp1d(santos_data_avg['time'], santos_data_avg[f'delnu{idx}_error'], bounds_error=False)
        pbobs = fint(time_arr)
        pbobse = finte(time_arr)
        domega_muhz_santos.append(pbobs)
        domega_err_santos.append(pbobse)

    sig_ratio = domega_err_santos/domega_sig[:, None]
    sig_ratio_mean = np.mean(sig_ratio, axis=1)
    sig_ratio_std = np.std(sig_ratio, axis=1)
    print(f" Sigma ratio: mean = {sig_ratio_mean} ± {sig_ratio_std}")
    fig, axs = plot_compare(time_arr, domega_muhz_santos, domega_err_santos, domega1muhz, domega_sig)
    fig.savefig(f'{papers_dir}/delnu1-comparison-{kicstr}.png')
    plt.close(fig)

    fig, axs = plot_compare(time_arr, domega_muhz_santos, domega_err_santos, domega2muhz, domega_sig)
    fig.savefig(f'{papers_dir}/delnu2-comparison-{kicstr}.png')
    plt.close(fig)

    fig, axs = plot_compare(time_arr, domega_muhz_santos, domega_err_santos, domega3muhz, domega_sig)
    fig.savefig(f'{papers_dir}/delnu3-comparison-{kicstr}.png')
    plt.close(fig)

    fig, axs = plot_cc(pschunks, pfilt_list, pexcl_list, time_arr, fittype='polynomial', dfreq=dfreq)
    fig.savefig(f'{papers_dir}/ccfit-{kicstr}.png')
    plt.close(fig)
