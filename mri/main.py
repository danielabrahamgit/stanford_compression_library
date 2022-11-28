# Import libraries
import sigpy as sp
import sigpy.mri as mr
import numpy as np
import os

# Plotting
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 18}
matplotlib.rc('font', **font)

from readout_compress import readout_compress
from data_loader import data_loader

# Set threads
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# Use GPU
device = sp.Device(0)
print(device)

# Load data
dl = data_loader('matlab_sim_2d')
img_consts, trj, dcf, ksp, mps = dl.load_dataset()
n = img_consts['n']

# Time compressors
compressors = [
    'Fully Sampled',
    'FS Noisy',
    'Cubic',
    'TVLPF',
    'Zero Order'
]

# Recon options
params = {
    'proxg': None, 
    'lamda': 1e-14,
    'accelerate': True,
    'max_iter': 20
}

# More params
rng = np.random.default_rng(100)
# rng = np.random
sigma = 1.0e-5
# sigma = 0.005
alpha = 1
cutoff = 0.01
dkmax = 1.5
recons = {}
tc_params = {}

# Add Noise
noise = rng.normal(0, sigma, ksp.shape) + rng.normal(0, sigma, ksp.shape) * 1j
ksp_noisy = ksp + noise

# Reconstruct the compressed data
tc = readout_compress(trj, dcf, alpha=alpha, cutoff=cutoff, dkmax=dkmax, device=device, interp_kind='linear')

for i, comp_type in enumerate(compressors):

    # compress to a file and get file size
    filename = f'data/{comp_type}.mrc'
    if comp_type == 'Fully Sampled':
        tc.encode_file(filename, comp_type, ksp)
        file_size = os.path.getsize(filename)
        file_size_full = file_size
    else:
        tc.encode_file(filename, comp_type, ksp_noisy)
        file_size = os.path.getsize(filename)

    # Decode compressed file
    ksp_tc, trj_tc, dcf_tc = tc.decode_file(filename, mps.shape[1:])

    # Compression ratio
    Rt = file_size_full / file_size

    # Reconstruct compressed data
    recon_tc = mr.app.SenseRecon(
        ksp_tc, 
        mps, 
        coord=trj_tc, 
        weights=dcf_tc, 
        device=device,
        lamda=params['lamda'],
        max_iter=params['max_iter'],
        show_pbar=False).run().get()
    
    # Parameters that define this time compression
    tc_param = np.array([alpha, cutoff, dkmax, Rt])

    # Save to dict
    if comp_type in recons:
        recons[comp_type].append(recon_tc)
    else:
        recons[comp_type] = [recon_tc]
    if comp_type in tc_params:
        tc_params[comp_type].append(tc_param)
    else:
        tc_params[comp_type] = [tc_param]

# Save to files
for comp_type in compressors:
    recon_array = np.array(recons[comp_type])
    tc_param_array = np.array(tc_params[comp_type])
    dl.recon_to_file(recon_array, tc_param_array, comp_type)
