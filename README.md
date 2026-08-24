# seismo-xl

<p align="center">
  <a href="https://pypi.org/project/seismo-xl/">
    <img src="https://img.shields.io/pypi/v/seismo-xl.svg?style=flat-square" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/seismo-xl/">
    <img src="https://img.shields.io/pypi/pyversions/seismo-xl.svg?style=flat-square" alt="Python versions">
  </a>
  <a href="https://github.com/samarth-kashyap/seismo-xl/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/seismo-xl.svg?style=flat-square" alt="License">
  </a>
  <a href="https://pypi.org/project/seismo-xl/">
    <img src="https://img.shields.io/pypi/dm/seismo-xl.svg?style=flat-square" alt="PyPI downloads">
  </a>
</p>

**Measure frequency shifts in solar-like oscillators using filtered cross-correlation.**

This package implements the filtered cross-correlation method for computing variations in p-mode frequencies ($\delta\omega_\ell$) over time. It supports both Kepler and Virgo/SoHO data.

## Method

The pipeline:

1. **Peakbagging** — Fit mode parameters on the time-averaged power spectrum using [apollinaire](https://gitlab.com/sybreton/apollinaire).
2. **Power spectrum construction** — Build a model power spectrum from Lorentzian profiles + Harvey-like background + photon noise.
3. **MI filter construction** — For each spherical harmonic degree $\ell$, construct a *mode-isolation (MI) filter*.
4. **Cross-correlation** — Cross-correlate each chunk's power spectrum with the MI filter. The peak shift gives $\delta\omega_\ell$.
5. **Uncertainty** — Monte Carlo realizations (noisify spectrum with $\chi^2_2$ distribution, re-run cross-correlation).

## Installation

### Prerequisites

- Python ≥ 3.9
- [uv](https://docs.astral.sh/uv/) (recommended) or pip + venv
- [apollinaire](https://gitlab.com/sybreton/apollinaire) — required for peakbagging

### Using uv (recommended)

```bash
# Create virtual environment with Python 3.13
uv venv --python 3.13

# Activate
source .venv/bin/activate

# Install seismo-xl in editable mode
uv sync

# With peakbagging + notebooks (apollinaire, jupyter):
uv sync --all-extras
```

### Using pip + venv

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# With peakbagging:
pip install -e ".[all]"
```

## Usage

### 1. Configuration

Edit `config.yml` (Kepler) or `config_virgo.yml` (Virgo/SoHO):

```yaml
Navg: 180            # Sub-series length (days)
Nshift: 45           # Shift between sub-series (days)
Nmcmc: 10000         # MCMC iterations for peakbagging
nmin: 16             # Minimum radial order
nmax: 26             # Maximum radial order
freqmin: 150.        # Min frequency (μHz)
freqmax: 6000.       # Max frequency (μHz)
data_dir: "./data"   # Lightcurve directory
output_dir: "/path/to/output"
```

### 2. Peakbagging

Fit mode parameters on the average spectrum:

```bash
# Kepler
python peakbag_kepler.py --kic 8006161 --peakbag

# Virgo/SoHO
python peakbag_virgo.py --peakbag
```

### 3. Compute frequency shifts

```bash
python compute_delnu.py --kic 8006161 --lmax 3
```

### 4. Synthetic tests (Virgo/SoHO mode set)

```bash
python create_spectra.py --source valeriy --channel blue --Ncarr 3
```

## Project structure

```
.
├── pyproject.toml          # Package metadata & dependencies
├── setup.py                # Legacy setup (editable install support)
├── src/
│   └── seismo_xl/
│       ├── __init__.py
│       ├── config.py           # YAML config loader with !include support
│       ├── globalvars.py       # Global constants & helper classes
│       ├── logger.py           # Logging utilities
│       ├── stellarspec.py      # Stellar power spectrum construction
│       └── utils.py            # Mode parameter readers & utilities
├── peakbag_kepler.py       # Kepler peakbagging entry point
├── peakbag_virgo.py        # Virgo/SoHO peakbagging entry point
├── compute_delnu.py        # Frequency shift computation
├── create_spectra.py       # Synthetic spectra (Virgo mode set)
├── config.yml              # Kepler configuration
├── config_virgo.yml        # Virgo/SoHO configuration
├── notebooks/              # Jupyter notebooks for analysis & validation
└── jobscripts/             # Slurm batch scripts
```

## Dependencies

| Package | Role |
|---|---|
| `numpy`, `scipy` | Numerical |
| `matplotlib` | Plotting |
| `h5py` | HDF5 I/O |
| `pandas` | CSV data |
| `pyyaml` | Config |
| `astropy` | FITS reading |
| `tqdm` | Progress bars |
| `pyshtools` | Legendre polynomials |
| `ritzLavelyPy` | Rotational splitting polynomials |
| `apollinaire` | Peakbagging (optional, needed for `--peakbag`) |

## Notes

### Configurable paths

All hardcoded scratch paths have been replaced with configurable CLI arguments or the `.config` file.

**Script-level overrides** (command-line args):

| Script | Flag | Default | Description |
|---|---|---|---|
| `compute_delnu.py` | `--output-dir` | `config.yml` → `output_dir` | Base dir for processed data |
| `compute_delnu.py` | `--papers-dir` | `{output_dir}/papers` | Where paper figures are saved |
| `create_spectra.py` | `--scratch-dir` | `/path/to/sun-intg` | Base dir for synthetic spectra |
| `create_spectra.py` | `--obs-dir` | *(required)* | Dir with `Larson_Schou_MDI_2015.dat` for ref mode params |

**Package-level paths** (`.config` file in repo root, one path per line):

```
# Line 0: eigenfunction directory (e.g., efs_Jesper)
# Line 1: Virgo/SoHO time-series data
# Line 2: processed output base
# Line 3: FWHM observations (contains Larson_Schou_MDI_2015.dat)
# Line 4: sun-integral processed/models
# Line 5: synthetics output
```

Create `.config` with your own paths before using `globalvars.py` features.

## Citation

If you use this code in published research, please cite:

> Kashyap, S. G., et al. (2026). *Determining low-ℓ p-mode frequency shifts in Sun-like stars: Enhancing the cross-correlation technique with filters*.

## License

MIT
