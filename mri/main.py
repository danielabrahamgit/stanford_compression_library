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

PROJ_DIR = '/local_mount/space/tiger/1/users/abrahamd/stanford_compression_library/mri'

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
    'TVLPF',
    'Zero Order'
]


# Store results here
recons = {}
comp_params = {}

# Fix noise seed and std
rng = np.random.default_rng(100)
sigma = 1e-5

# TVLPF Params
dkmax = 1.5

# Residual Quantization Params
ksp_skip = 20
bits_resid = 4

# Trajectory Compression Params
trj_start = 300
trj_skip  = 15

# Sweep a particular param
params = [0]

for k, param in enumerate(params):

    # Set param
    # trj_skip = param

    # Add Noise
    noise = rng.normal(0, sigma, ksp.shape) + rng.normal(0, sigma, ksp.shape) * 1j
    ksp_noisy = ksp + noise

    # Reconstruct the compressed data
    tc = readout_compress(trj, dcf, alpha=1, cutoff=0.03, dkmax=dkmax, device=device)

    for i, comp_type in enumerate(compressors):

        # Name of file depends on compressor type
        filename = f'{PROJ_DIR}/data/{comp_type}.mrc'

        # No Compression No Noise
        if comp_type == 'Fully Sampled':
            if k ==0:
                tc.encode_file(
                    filename, 
                    comp_type, 
                    ksp, 
                    trj_start=0, 
                    trj_skip=1,
                    ksp_skip=1,
                    bits_resid=0,
                    use_lossless=False)
            file_size = os.path.getsize(filename)
            file_size_full = file_size
        # No Compression + Noise
        elif comp_type == 'FS Noisy':
            if k == 0:
                tc.encode_file(
                    filename, 
                    comp_type, 
                    ksp_noisy, 
                    trj_start=0, 
                    trj_skip=1,
                    ksp_skip=1,
                    bits_resid=0,
                    use_lossless=False)
            file_size = os.path.getsize(filename)
        # Compression + Noise
        else:
            tc.encode_file(
                filename, 
                comp_type, 
                ksp_noisy, 
                trj_start=trj_start, 
                trj_skip=trj_skip,
                ksp_skip=ksp_skip,
                bits_resid=bits_resid,
                use_lossless=True)
            file_size = os.path.getsize(filename)

        # Reconstruct compressed data
        if comp_type != 'Fully Sampled' and comp_type != 'FS Noisy':
            # Decode compressed file
            ksp_tc, trj_tc, dcf_tc = tc.decode_file(filename)
            recon_tc = mr.app.SenseRecon(
                ksp_tc, 
                mps, 
                coord=trj_tc, 
                weights=dcf_tc, 
                device=device,
                lamda=0,
                max_iter=10,
                show_pbar=False).run().get()
        else:
            if k == 0:
                # Decode compressed file
                ksp_tc, trj_tc, dcf_tc = tc.decode_file(filename)
                recon_tc = mr.app.SenseRecon(
                ksp_tc, 
                mps, 
                coord=trj_tc, 
                weights=dcf_tc, 
                device=device,
                lamda=0,
                max_iter=10,
                show_pbar=False).run().get()
            else:
                recon_tc = recons[comp_type][0]
        
        # Parameters of this compressor
        comp_param = np.array([
            dkmax, 
            ksp_skip, 
            bits_resid, 
            trj_skip, 
            trj_start, 
            file_size,
            file_size_full / file_size])

        # Save to dict
        if comp_type in recons:
            recons[comp_type].append(recon_tc)
        else:
            recons[comp_type] = [recon_tc]
        if comp_type in comp_params:
            comp_params[comp_type].append(comp_param)
        else:
            comp_params[comp_type] = [comp_param]

# Save to files
for comp_type in compressors:
    recon_array = np.array(recons[comp_type])
    tc_param_array = np.array(comp_params[comp_type])
    dl.recon_to_file(recon_array, tc_param_array, comp_type)
