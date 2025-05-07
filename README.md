# Filtered cross-correlation method for computing frequency changes

This package enables the computing of variation in p-mode frequencies $\delta\omega_\ell$ over time using cross-correlation with filters. 

## Example

The first step is to obtain mode parameters using peak bagging. Peakbagging is first performed on averaged spectra. The configuration is specifed in `config.yml`. The contents of the config file are given below

```
Navg: 180
Nshift: 45
Nmcmc: 10000
nmin: 16
nmax: 26
freqmin: 150.
freqmax: 6000.
data_dir: "./data"
output_dir: "/scratch/seismo/kashyap/processed/p11-seismo-xl"
```
`Navg` - is the length of sub-series (days)
`Nshift` - difference in start times between adjacent sub-series (days)
`Nmcmc` - Number of MCMC iterations needed for computing errors
`nmin` - Minimum radial order for peakbagging
`nmax` - Maximum radial order for peakbagging
`freqmin` - Minimum frequency for peakbagging (muHz)
`freqmax` - Maximum frequency for peakbagging (muHz)
`data_dir` - Path of lightcurves
`output_dir` - Path of output files

After setting up config file, first run peakbagging using 
```python peakbag_kepler.py --kic 8006161```
Note that this step requires the use of the [apollinaire](https://gitlab.com/sybreton/apollinaire) package.

Once the peakbagging is complete. You can compute frequency changes using
```python compute_delnu.py --kic 8006161```

