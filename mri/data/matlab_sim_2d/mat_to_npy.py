import numpy as np
from scipy.io import loadmat

# File io
PROJ_DIR = './'
data = loadmat(PROJ_DIR + 'data/matlab_sim_2d/Sim_data_2d_for_Daniel.mat')

# Data has 3 spiral readouts in 2d. Let's just take one of these readouts
ro_inds = np.array([0, 1, 2])
dcf = np.moveaxis(data['DCF_ii'][0, :, ro_inds], 0, -1)
coord = np.moveaxis(data['crds_ii'][:2, :, ], 0, -1)[:, ro_inds, :]
ksp = np.moveaxis(data['kspace_signal_frame'][0, :, :, :], -1, 0)[:, :, ro_inds]
mps = np.moveaxis(data['sens'][:, :, 0, :], -1, 0)

# save
data_dir = PROJ_DIR + 'data/matlab_sim_2d/'
np.save(data_dir + 'dcf.npy', dcf)
np.save(data_dir + 'trj.npy', coord)
np.save(data_dir + 'ksp.npy', ksp)
np.save(data_dir + 'mps.npy', mps)
print('Saved data, exiting')