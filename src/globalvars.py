import numpy as np
import logging
import os

# Local imports
from . import logger

# Creating logger
LOGGER = logger.create_logger_stream(__name__, logging.NOTSET)

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()


class dirConfig():
    """Class where the director structure of package is listed"""
    with open(f"{package_dir}/.config", "r") as f:
        dirnames = f.read().splitlines()

    # from .config
    # [0] /scratch/seismo/kashyap/codes/get-solar-eigs/efs_Jesper
    # [1] /scratch/seismo/kashyap/data/virgo
    # [2] /scratch/seismo/kashyap/processed
    # [3] /scratch/seismo/kashyap/data
    # [4] /scratch/seismo/kashyap/processed/sun-intg
    # [5] /scratch/seismo/kashyap/synthetics/sun-intg

    # directories related to eigenfunctions
    efsdir = dirnames[0]
    data_dir = f"{efsdir}/snrnmais_files/data_files"
    eigs_dir = f"{efsdir}/snrnmais_files/eig_files"

    package_dir = f"{package_dir}"
    obs_ts_dir = dirnames[1]
    processed_dir = dirnames[2]
    obs_dir = dirnames[3]
    save_dir = dirnames[4]
    models_dir = dirnames[4]
    synth_dir = dirnames[5]


class defaultArgs():
    def __init__(self):
        self.source = "valeriy"
        self.channel = "blue"
        self.Ncarr = 3
        self.chunksize = self.Ncarr*25.38
        self.skipmax = 40
        self.lmax = 3
        self.inclang = 90
        self.alpha = 1.0
        self.beta = 0.0
        self.butter = 1.5
        self.ellfilter = 0

    def update_args(self, newargs):
        if newargs is not None:
            newkeys = vars(newargs).keys()
            for key in newkeys:
                if key in vars(self).keys():
                    vars(self)[key] = vars(newargs)[key]


class writeDirConfig():
    """Class to deal with directory config of output files"""
    def __init__(self, ARGSip=None, processed_dir='/scratch/seismo/kashyap/processed/sun-intg'):
        """
        --- critical parameters ---
        ARGS.source    -- (data, model)
        ARGS.channel   -- (data, model)
        ARGS.Ncarr     -- (data, model)
        ARGS.skipmax   -- (data, model)
        ARGS.lmax      -- (model)
        ARGS.inclang   -- (model)

        --- optional parameters ----
        ARGS.alpha   -- (model)
        ARGS.beta    -- (model)
        ARGS.butter  -- (model)
        """
        ARGS = defaultArgs()
        ARGS.update_args(ARGSip)
        DATASFX0 = f"{ARGS.source}-{ARGS.channel}-Ncarr{ARGS.Ncarr}-skip{ARGS.skipmax}"
        MODELSFX0 = f"{DATASFX0}-ell{ARGS.lmax}-i{ARGS.inclang:02d}"
        if ARGS.ellfilter>0:
            DATASFX = DATASFX0 + f"-l{ARGS.ellfilter}flt"
            MODELSFX = MODELSFX0 + f"-l{ARGS.ellfilter}flt"
        else:
            DATASFX = DATASFX0
            MODELSFX = MODELSFX0
        self.processed_dir = processed_dir
        self.ddir = f"{self.processed_dir}/data/{DATASFX}"
        self.mdir = f"{self.processed_dir}/model/{MODELSFX}"
        self.fitsdir = f"{self.processed_dir}/ps-fits/{MODELSFX0}"
        self.mkerndir = f"{self.processed_dir}/model/{MODELSFX}/kernels"
        self.dkerndir = f"{self.processed_dir}/data/{DATASFX}/kernels"
        self.mfile_sfx = f"a{ARGS.alpha:3.1f}-b{ARGS.beta:3.1f}-bw{ARGS.butter:3.1f}"
        self.dfile_sfx = f"bw{ARGS.butter:3.1f}"

        self.pd_dir = f"{self.processed_dir}/plots/data-{DATASFX}"
        self.pm_dir = f"{self.processed_dir}/plots/model-{MODELSFX}"
        self.magdir = (f"{self.processed_dir}/magnetograms/Ncarr{ARGS.Ncarr}")



class globalVars():
    T = 2678460.
    dt = 60.
    Nt = int(T/dt)
    homega = 2.*np.pi/T
    omegaPlus = np.arange((Nt+1)/2)*homega
    nuPlus = omegaPlus / (2.*np.pi)
    obs_dir = '/home/fournier/mps_montjoie/data/Observations/FWHM_OBS'
    efs_dir = dirnames[0]
    obs_ts_dir = dirnames[1]
    LOGGER.info("Successfully loaded globalVars")

    def __init__(self):
        self.dt = 60.
        self.package_dir = package_dir


    def load_data(self, dataset='larson', lmax=3):
        DIRS = dirConfig()
        data = np.genfromtxt(f'{self.obs_dir}/Larson_Schou_MDI_2015.dat')
        ells = data[:, 0].astype('int')
        enns = data[:, 1].astype('int')
        nu  = data[:, 2]*1e-6      # w/2pi in Hz
        fwhm = data[:, 4]*1e-6     # FWHM in Hz
        sig_fwhm = data[:, 10]     # standard deviation on the FWHM
        if dataset=='larson':
            return ells, enns, nu, fwhm, sig_fwhm
        elif dataset=='refined':
            ELLS = np.load(f"{DIRS.save_dir}/ps-fits/fitted-ell-list-{lmax}.npy").astype('int')
            ENNS = np.load(f"{DIRS.save_dir}/ps-fits/fitted-enn-list-{lmax}.npy").astype('int')
            NUS = np.load(f"{DIRS.save_dir}/ps-fits/fitted-nu-list-mod-{lmax}.npy")
            FWHMS = np.load(f"{DIRS.save_dir}/ps-fits/fitted-fwhm-list-mod-{lmax}.npy")
            SIG_FWHMS = []
            for idx in range(len(ELLS)):
                _ell, _enn = ELLS[idx], ENNS[idx]
                try:
                    mode_idx = np.where((ells==_ell)*(enns==_enn))[0][0]
                    sigfw = sig_fwhm[mode_idx]
                except IndexError:
                    pass
                SIG_FWHMS.append(sigfw)
            SIG_FWHMS = np.array(SIG_FWHMS)
            return ELLS, ENNS, NUS, FWHMS, SIG_FWHMS


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
        npoly = [(enn - enn0)**i for i in range(len(gi))]
        gamma_nl = npoly @ gi
        return gamma_nl



    def gapfilled_nl(self, lmax=3, nmax=31):
        ells, enns, nu, fwhm, sig_fwhm = self.load_data()
        ell_list = []
        enn_list = []
        nu_list = []
        fwhm_list = []
        
        for ell in range(lmax):
            mask_ell = ells==ell
            enn_ell = enns[mask_ell]
            nu_ell = nu[mask_ell]
            fwhm_ell = fwhm[mask_ell]

            for enn in range(int(enn_ell.min()), nmax, 1):
                nexists = (enn_ell==enn).sum()
                if nexists > 0:
                    idx = np.where(enn_ell==enn)[0][0]
                    ell_list.append(ell)
                    enn_list.append(enn_ell[idx])
                    nu_list.append(nu_ell[idx])
                    fwhm_list.append(fwhm_ell[idx])
                else:
                    nu_nl = self.get_nunl_stahn(enn, ell)*1e-6       #Hz
                    gamma_nl = self.get_gammanl_stahn(enn, ell)*1e-6 #Hz
                    ell_list.append(ell)
                    enn_list.append(enn)
                    nu_list.append(nu_nl)
                    fwhm_list.append(gamma_nl)
        return ell_list, enn_list, nu_list, fwhm_list
