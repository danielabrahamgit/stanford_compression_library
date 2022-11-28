# Import libraries
import numpy as np
import sigpy as sp

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from plotting_util import plotting_util
from data_loader import data_loader
from readout_compress import readout_compress

# Set threads
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# Use GPU
device = sp.Device(-1)
print(device)

# Load data
dl = data_loader('matlab_sim_2d')
img_consts, trj, dcf, ksp, mps = dl.load_dataset()


# Time compressors
compressors = [
    'Fully Sampled',
    'FS Noisy',
    'Cubic',
    'TVLPF',
    'Zero Order'
]

# Store reconstructions and time compression parameters into dicts
recons = {}
tc_params = {} 

for i, comp_type in enumerate(compressors):
    recons[comp_type], tc_params[comp_type] = dl.recon_from_file(comp_type)

# Plot with plotting util
pu = plotting_util(recons, tc_params, dt=img_consts['dt'], figsize=(14, 7))
pu.recon_figure()
# pu.gfactor_error_animation(file_gif)
# pu.gfactor_error_figure()
# pu.tc_plot_figure()
plt.show()
