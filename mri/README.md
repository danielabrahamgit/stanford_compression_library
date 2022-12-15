## Setup
Please follow the instructions in the SCL repo for setup.    
In addition, you will need the sigpy library to reconstruct MRI images. The instructions can be found here:
https://sigpy.readthedocs.io/en/latest/index.html

## Using The Code
The `main.py` file is responsible for using the compressor (located in `readout_compress.py`). `main.py` will run the compressor (with a set of parameters), save it to a file, read the file, and write the reconstructed image to a new file for plotting. If you want to skip all this and just want the simple encode decode functionality:
```
import sigpy as sp
from readout_compress import readout_compress

# TVLPF Params
dkmax = 1.5

# Residual Quantization Params
ksp_skip = 20
bits_resid = 4

# Trajectory Compression Params
trj_start = 300
trj_skip  = 15

# ENCODE
tc = readout_compress(trj, dcf, alpha=1, cutoff=0.03, dkmax=dkmax)
tc.encode_file(filename, 
               comp_type, 
               ksp, 
               trj_start, 
               trj_skip,
               ksp_skip,
               bits_resid)

# DECODE
ksp, trj, dcf = tc.decode_file(filename)
```
The parameters `dkmax, ksp_skip, bits_resid, trj_start, trj_skip` are all explained in `final.md`.   
The actual MRI data is given the frequency domain data: `ksp`, the frequency domain coordinates: `trj`, and the density compensation factor: `dcf`. 

## Reproducibility
To reproduce the final image shown in the report below, run:
```
python main.py && python generate_plots.py
```

## Useful Resources
Slides: https://docs.google.com/presentation/d/11L9BWN0yfgqKCR806ICHXbua8WU5ksKaRbGLSgsoFxk/edit?usp=sharing.  
Report: https://github.com/danielabrahamgit/stanford_compression_library/blob/main/mri/final.md
