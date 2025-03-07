import apollinaire as apn
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

if __name__ == "__main__":
    hdu = fits.open('data/kplr008006161_kasoc-ts_slc_v1.fits')[1]
    data = np.array (hdu.data)
    t = data['TIME']
    v = data['FLUX']
    masknan = np.isnan(v)
    v[masknan] = 0.
    # 
    # fig, ax = plt.subplots ()
    # ax.plot (t-t[0], v, color='black')
    # ax.set_xlabel ('Time (days)')
    # ax.set_ylabel ('Luminosity variation (ppm)')
    # 
    dt = np.median (t[1:] - t[:-1]) * 86400
    freq, psd = apn.psd.series_to_psd(v, dt=dt, correct_dc=True)
    freq = freq*1e6
    psd = psd*1e-6
    cond = (freq>1000.)&(freq<6000.)

    fig, ax = plt.subplots ()
    ax.plot (freq[cond], psd[cond], color='black')
    ax.set_xlabel (r'Frequency ($\mu$Hz)')
    ax.set_ylabel (r'PSD (ppm$^2$ / $\mu$Hz)')
    dnu = 149.4
    r, m, teff = 0.931, 0.990, 5488
    ed = apn.psd.echelle_diagram(freq[cond], psd[cond], dnu, smooth=100,
                                cmap='Blues', shading='gouraud', vmax=3,
                                figsize=(8,6))

    order_to_fit = np.arange(10) + 16
    apn.peakbagging.stellar_framework(freq[cond], psd[cond], r, m, teff, 
                                      n_harvey=2, low_cut=50., 
                                      filename_back='background.png', 
                                      filemcmc_back='mcmc_background.h5', 
                                      nsteps_mcmc_back=2000, discard_back=1500,
                                      n_order=6, n_order_peakbagging=11,
                                      filename_pattern='pattern.png', fit_l3=True,
                                      filemcmc_pattern='mcmc_pattern.h5', 
                                      nsteps_mcmc_pattern=5000, parallelise=True,
                                      quickfit=False, discard_pkb=1000, progress=True, nwalkers=50,
                                      a2z_file='modes_param.a2z', format_cornerplot='png', 
                                      nsteps_mcmc_peakbagging=5000,
                                      filename_peakbagging='summary_peakbagging.png', 
                                      dpi=300)