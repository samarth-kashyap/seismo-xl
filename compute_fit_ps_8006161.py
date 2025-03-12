import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import apollinaire as apn
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument('--Navg', type=int, default=90 ,help='Length of sub-series (days)')
parser.add_argument('--Nshift', type=int, default=15, help='Shift between sub-series (days)')
parser.add_argument('--peakbag', action='store_true', help='Fit spectra')
parser.add_argument('--Nmcmc', type=int, default=10000, help='Num of MCMC steps')
ARGS = parser.parse_args()

data_dir = "/data/seismo/kashyap/codes/p11-seismo-xl/data"


def gaussian(x, mu, fwhm):
    """Returns a gaussian of chosen center and fwhm.

    Parameters
    ----------
    :x: range over which gaussian is defined
    :type: np.ndarray(ndim=1, dtype=float)

    :mu: center of the gaussian
    :type: float

    :fwhm: FWHM of the gaussian
    :type: float

    Returns
    -------
    :gn: gaussian profile
    :type: np.ndarray(ndim=1, dtype=float)
    """
    sigma = fwhm / np.sqrt(8. * np.log(2.))
    gn = np.exp(-(x-mu)**2/2./sigma/sigma)
    return gn




if __name__ == "__main__":
    ts = fits.open(f'{data_dir}/kplr008006161_kasoc-ts_slc_v1.fits')
    tsdata = ts[1].data
    time = tsdata['TIME']
    flux = tsdata['FLUX']
    masknan = np.isnan(flux)
    time = time[~masknan]
    flux = flux[~masknan]

    kepler_obs_start = 54965 # BKJD corresponding to May 13, 2009 (operation start of Kepler)
    time = time - kepler_obs_start
    time_min = time[0]
    time_max = time[-1]
    dt = np.diff(time)
    dt_uni = np.median(dt)
    num_uni = int((time_max - time_min)/dt_uni)
    time_uni = np.arange(num_uni)*dt_uni + time_min

    f1d = interp1d(time, flux, fill_value=0., bounds_error=False)
    flux_uni = f1d(time_uni)
    freq = np.fft.rfftfreq(len(time_uni), d=dt_uni*24*3600.)
    powr = abs(np.fft.rfft(flux_uni))**2

    day2sec = 24*3600.
    filter_mu = 3.616*1e-3
    filter_fwhm = 1.5e-3

    obs_days = time_uni[-1] - time_uni[0]
    num_chunks = int(obs_days//ARGS.Nshift)

    avg_idx = int(ARGS.Navg/(dt_uni))
    if avg_idx%2==1: avg_idx += 1
    shf_idx = int(ARGS.Nshift/(dt_uni))
    tstart_idx = 0
    time_list = []
    tmid_list = []
    flux_list = []
    pow_list = []
    pow_list_gfilter = []
    freq_list = []
    ct_list = []
    for idx in range(num_chunks):
        tend_idx = tstart_idx + avg_idx
        flux_chunk = flux_uni[tstart_idx:tend_idx]
        time_chunk = time_uni[tstart_idx:tend_idx]
        if len(time_chunk) < avg_idx:
            break
        time_list.append(time_chunk)
        flux_list.append(flux_chunk)
        tmid_list.append(time_chunk.mean())
        _fluxfft = np.fft.rfft(flux_chunk)
        _freq = np.fft.rfftfreq(len(time_chunk), d=dt_uni*day2sec)
        sigpower = abs(_fluxfft)**2
        gfilter = gaussian(_freq, filter_mu, filter_fwhm)
        pow_list.append(sigpower)
        pow_list_gfilter.append(sigpower*gfilter)
        tstart_idx += shf_idx

    freq_ref = _freq*1.
    pow_list = np.array(pow_list)
    pow_list_gfilter = np.array(pow_list_gfilter)
    pow_gfilter_ref = pow_list_gfilter.mean(axis=0)
    pow_ref = pow_list.mean(axis=0)
    pshapelist = [pow.shape[0] for pow in pow_list]
    print(f"Shape of power spectrum = {np.unique(pshapelist)}")
    outputdir = f"/data/seismo/kashyap/codes/p11-seismo-xl/results/N{ARGS.Navg}-s{ARGS.Nshift}"
    os.system(f'mkdir {outputdir}')
    np.save(f"{data_dir}/pschunks-8006161-N{int(ARGS.Navg)}-s{int(ARGS.Nshift)}.npy", pow_list)
    np.save(f"{data_dir}/freq-8006161-N{int(ARGS.Navg)}-s{int(ARGS.Nshift)}.npy", freq_ref)
    np.save(f"{data_dir}/tmid-8006161-N{int(ARGS.Navg)}-s{int(ARGS.Nshift)}.npy", np.array(tmid_list))

    freq_muhz = freq_ref*1e6
    psd = pow_list.mean(axis=0)*1e-9
    cond = (freq_muhz>150.)*(freq_muhz<6000.)
    print(f"Number of freq bins = {cond.sum()}")

    fig, ax = plt.subplots()
    ax.plot (freq_muhz[cond], psd[cond], color='black')
    ax.set_xlabel (r'Frequency ($\mu$Hz)')
    ax.set_ylabel (r'PSD (ppm$^2$/$\mu$Hz)')
    fig.tight_layout()
    fig.savefig(f"{outputdir}/power_spectrum.png")
    dnu = 149.4
    r, m, teff = 0.931, 0.990, 5488
    ed = apn.psd.echelle_diagram(freq_muhz[cond], psd[cond], dnu, smooth=100,
                                cmap='Blues', shading='gouraud', vmax=0.5,
                                figsize=(8,6))

    order_to_fit = np.arange(10) + 16
    if ARGS.peakbag:
        print('Begin peakbagging')
        apn.peakbagging.stellar_framework(freq_muhz[cond], psd[cond], r, m, teff, 
                                        n_harvey=2, low_cut=50., dpi=300,
                                        filename_back=f'{outputdir}/background.png',
                                        filemcmc_back=f'{outputdir}/mcmc_background.h5',nsteps_mcmc_back=ARGS.Nmcmc, 
                                        discard_back=int(0.75*ARGS.Nmcmc),
                                        n_order=6, n_order_peakbagging=11,
                                        filename_pattern=f'{outputdir}/pattern.png', fit_l3=True,
                                        filemcmc_pattern=f'{outputdir}/mcmc_pattern.h5',
                                        nsteps_mcmc_pattern=ARGS.Nmcmc, parallelise=True,
                                        mcmcDir=outputdir,
                                        quickfit=False, 
                                        discard_pkb=int(0.75*ARGS.Nmcmc), 
                                        progress=True,
                                        nwalkers=50, 
                                        a2z_file=f'{outputdir}/modes_param.a2z',
                                        format_cornerplot='png', 
                                        nsteps_mcmc_peakbagging=ARGS.Nmcmc,
                                        filename_peakbagging=f'{outputdir}/summary_peakbag.png',)
