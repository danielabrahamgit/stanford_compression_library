# Import libraries
import numpy as np
import sigpy as sp

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

# Plot fonts
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 17}
matplotlib.rc('font', **font)

from data_loader import data_loader

# Set threads
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

def normalize(shifted, target):
    col1 = shifted.flatten()
    col2 = col1 * 0 + 1
    A = np.array([col1, col2]).T
    y = target.flatten()
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a * shifted + b

def rate_distortion_figure(recons, comp_params, figsize=(14, 7)):

    plt.figure(figsize=figsize)
    param_name = ''
    params = None

    for i, comp_type in enumerate(recons.keys()):

        # Get params
        dkmaxs = comp_params[comp_type][:, 0]
        ksp_skips = comp_params[comp_type][:, 1]
        bits_resids = comp_params[comp_type][:, 2]
        trj_skips = comp_params[comp_type][:, 3]
        trj_starts = comp_params[comp_type][:, 4]
        file_sizes = comp_params[comp_type][:, 5]
        comp_ratios = comp_params[comp_type][:, 6]

        # Find which param we are sweeping
        if param_name == '' and params is None:
            if dkmaxs[0] != dkmaxs[1]:
                param_name = 'dkmax'
                params = dkmaxs
            elif ksp_skips[0] != ksp_skips[1]:
                param_name = 'ksp_skip'
                params = ksp_skips
            elif bits_resids[0] != bits_resids[1]:
                param_name = 'bits_resid'
                params = bits_resids
            elif trj_skips[0] != trj_skips[1]:
                param_name = 'trj_skip'
                params = trj_skips
            elif trj_starts[0] != trj_starts[1]:
                param_name = 'trj_start'
                params = trj_starts
    
        # For each param, compute MSE
        mses = []
        rates = []
        for k in range(len(params)):
            recon_tc = recons[comp_type][k]
            if comp_type == 'Fully Sampled':
                recon_fs = recon_tc
            if comp_type == 'TVLPF' or comp_type == 'Zero Order':
                # Normalize
                recon_tc = normalize(recon_tc, recon_fs)
                err = np.abs(np.abs(recon_fs) - np.abs(recon_tc))
                MSE = np.mean(np.abs(err) ** 2, axis=(0, 1))
                mses.append(MSE)
                rates.append(file_sizes[k]  / 2 ** 10)
        if comp_type == 'TVLPF' or comp_type == 'Zero Order':
            plt.plot(rates, mses, label=comp_type)
        
    
    plt.title(param_name)
    plt.xlabel('File Size [kB]')
    plt.ylabel('MSE')
    plt.legend()
    
def recon_figure(recons, comp_params, figsize=(14, 7)):

    assert len(recons['Fully Sampled'].shape) == 2
    
    # Setup figure
    fig_recon, axs_recon = plt.subplots(nrows=2, 
                            ncols=len(recons), 
                            gridspec_kw={'wspace':0, 'hspace':0},
                            figsize=figsize,
                            squeeze=True)

    # Plot all
    for i, comp_type in enumerate(recons.keys()):
        
        # Get time compressed recon
        recon_tc = recons[comp_type]

        # Compute SNR 
        def snr_func(img):
                noise_std = np.std(img[89:128, 3:18].flatten())
                sig_strn = np.mean(np.abs(img[82:127, 82:127].flatten()))
                return sig_strn / noise_std
        SNR = snr_func(recon_tc)

        # Max/min from fully sampled
        if comp_type == 'Fully Sampled':
            recon_fs = recon_tc
            vmin = np.abs(recon_fs).min()
            vmax = np.abs(recon_fs).max()

        # Normalize
        recon_tc = normalize(recon_tc, recon_fs)

        # Recon figure
        err = np.abs(np.abs(recon_fs) - np.abs(recon_tc))
        MSE = np.mean(np.abs(err) ** 2, axis=(0, 1))
        if len(recons) == 1:
            ax_recon = axs_recon[0]
            ax_recon_diff = axs_recon[1]
        else:
            ax_recon = axs_recon[0, i]
            ax_recon_diff = axs_recon[1, i]
        ax_recon.set_title(comp_type)
        ax_recon.imshow(np.abs(recon_tc), cmap='gray', vmin=vmin, vmax=vmax)
        ax_recon.axis('off')
        ax_recon_diff.axis('off')
        ax_recon_diff.imshow(err, cmap='gray', vmin=0, vmax=vmax/5)

        # Print Stuff
        print(f'{comp_type:<20}SNR = {SNR:.2f}', end=', ')
        print(f'MSE = {MSE:.3e}', end=', ')
        print(f'Comp Ratio = {comp_params[comp_type][6]}')

# Use GPU
device = sp.Device(-1)
print(device)

# Load data
dl = data_loader('matlab_sim_2d')

# Time compressors
compressors = [
    'Fully Sampled',
    'FS Noisy',
    'TVLPF',
    'Zero Order'
]

# Store reconstructions and time compression parameters into dicts
recons = {}
comp_params = {} 
for i, comp_type in enumerate(compressors):
    recons[comp_type], comp_params[comp_type] = dl.recon_from_file(comp_type)

# Plot and save
# rate_distortion_figure(recons, comp_params)
recon_figure(recons, comp_params)
plt.savefig('plz2')
