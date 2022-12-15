import numpy as np
import os
from pathlib import Path

# PROJ_DIR = '/local_mount/space/tiger/1/users/abrahamd/mr_time_compression'
# PROJ_DIR = str(Path.home()) + '/devel/mr_time_compression'
# PROJ_DIR = os.getcwd()
PROJ_DIR = '/local_mount/space/tiger/1/users/abrahamd/stanford_compression_library/mri'

class data_loader:

    def __init__(self, dataset):
        self.dataset = dataset

    def recon_from_file(self, recon_name):
        recons = np.load(PROJ_DIR + f'/data/recon/{recon_name}_{self.dataset}.npz')['arr_0']
        tc_params = np.load(PROJ_DIR + f'/data/recon/{recon_name}_{self.dataset}.npz')['arr_1']
        if recons.shape[0] == 1:
            recons = recons[0]
        if tc_params.shape[0] == 1:
            tc_params = tc_params[0]
        return recons, tc_params
    
    def recon_to_file(self, recons, tc_params, recon_name):
        np.savez(PROJ_DIR + f'/data/recon/{recon_name}_{self.dataset}.npz', recons, tc_params)

    def load_dataset(self):

        # File io
        data_dir = PROJ_DIR + f'/data/{self.dataset}/'
        ksp = np.load(data_dir + 'ksp.npy')
        trj = np.load(data_dir + 'trj.npy')
        dcf = np.load(data_dir + 'dcf.npy')
        mps = np.load(data_dir + 'mps.npy')

        # Imaging Constants
        img_consts = {
            'R': 3,
            'dt':2e-6,
            'n': mps.shape[1],
            'fov': 220e-3 ,
            'res': 1e-3
        }

        # Print Stuff
        print(f'trj shape = {trj.shape}')
        print(f'ksp shape = {ksp.shape}')
        print(f'dcf shape = {dcf.shape}')
        print(f'mps shape = {mps.shape}')

        # Return stuff
        return img_consts, trj, dcf, ksp, mps