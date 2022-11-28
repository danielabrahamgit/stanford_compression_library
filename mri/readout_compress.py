import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

import numpy as np
import time
import sigpy as sp
import struct
import bitarray

from bitarray_utils import uint_to_bitarray, bitarray_to_uint
from sigpy.mri.dcf import pipe_menon_dcf
from scipy.interpolate import interp1d
from scipy.signal.windows import get_window

class readout_compress:

    comp_types = [
        'Fully Sampled',
        'FS Noisy',
        'Cubic', 
        'TVLPF',
        'Zero Order'
    ]

    def __init__(self, trj, dcf, alpha=1, cutoff=0, dkmax=0.95, interp_kind='linear', device=None):
        """
        A readout compression instance will compress along the k-space trajectory. 
        We will have many different ways to perform the compression, all of which
        are functions of this class. We will assume a constant dt between sampled 
        points.

        Inputs:
            trj - the k-space coordinate
                Shape = (a, b, c):
                    a - points along readout (we will compress this dimension)
                    b - number of interleaves or shots or TRs
                    c - dimension of data (2 for 2D acquisition, or 3 for 3D acquisition)
            dcf - Density compensation factor
                Shape = (a, b)
            alpha - 1 for uniform sampling, 0 for original trajectory (middle value for something in the middle)
            cutoff - 0 for no cutoff, 1 for keep all the same trajectory, middle for something in the middle
            dkmax - maximum k-space spacing
            device - Cuda device or CPU
        """

        # Make sure our data dimensionality is 
        assert trj.shape[2] < 3

        # Save
        self.trj = trj
        self.device = device
        self.dcf = dcf
        self.alpha = alpha
        self.cutoff = cutoff
        self.dkmax = dkmax
        self.interp_kind = interp_kind
        if device is None:
            self.device = sp.get_device(trj)
        else:
            self.device = device
        self.noise_cov_matrix = None
        
        # Resampling
        self.t_tc, self.trj_tc, self.dcf_tc, self.ind_start = self.resample_trajectory(interp_kind)

    def encode_file(self, filename, comp_type, ksp):
        
        # Compress spiral
        ksp_new, trj_new, _ = self.compress(comp_type, ksp)

        if comp_type == 'Fully Sampled' or comp_type == 'FS Noisy':
            ksp_tc, trj_tc = ksp_new, trj_new
            M = 1
            bits_resid = 0
            mn_resid = 0
            mx_resid = 0
        else:

            # Downsample by factor of M
            M = 4
            t = np.arange(trj_new.shape[0])
            trj_tc = trj_new[::M, :, :]
            ksp_tc = ksp_new[:, ::M, :]
            
            # Recon and find residuals
            ksp_est = interp1d(t[::M], ksp_tc, kind='cubic', axis=1, bounds_error=False, fill_value=0)(t)
            resid = ksp_est - ksp_new

            # Quanitzation
            bits_resid = 12
            mn_resid = min(resid.real.min(), resid.imag.min())
            mx_resid = max(resid.real.max(), resid.imag.max())
            levels = np.linspace(mn_resid, mx_resid, 2 ** bits_resid)
            resid_num_real = np.argmin(np.abs(resid[:, :, :, np.newaxis].real - levels[np.newaxis, np.newaxis, np.newaxis, :]), axis=-1)
            resid_num_imag = np.argmin(np.abs(resid[:, :, :, np.newaxis].imag - levels[np.newaxis, np.newaxis, np.newaxis, :]), axis=-1)

            # If interpolated point, compress residual difference (a lot)!
            resid_inds = t
            resid_inds = np.delete(resid_inds, t[::M])
            ksp_tc = ksp_new.copy()
            ksp_tc[:, resid_inds, :] = resid[:, resid_inds, :]

        # Save to binary file
        ba = bitarray.bitarray()
        
        def float_to_bitarray(flt_val):
            ba = bitarray.bitarray()
            flt = struct.pack('>f', flt_val)
            ba.frombytes(flt)
            return ba

        # Store dimensions first
        nr = ksp_new.shape[1]
        nc = ksp_new.shape[0]
        ba += uint_to_bitarray(nr, 32)
        ba += uint_to_bitarray(nc, 32)
        ba += uint_to_bitarray(M, 32)
        ba += uint_to_bitarray(bits_resid, 32)
        ba += float_to_bitarray(mn_resid)
        ba += float_to_bitarray(mx_resid)

        print(nr, nc, M, bits_resid, mn_resid, mx_resid)
        
        for r in range(nr):

            # Encode trajectory
            # if r % M == 0:
            ba += float_to_bitarray(trj_new[r, 0, 0])
            ba += float_to_bitarray(trj_new[r, 0, 1])

            # Encode all coils data
            for c in range(nc):

                if r % M == 0:
                    ba += float_to_bitarray(ksp_new[c, r, 0].real)
                    ba += float_to_bitarray(ksp_new[c, r, 0].imag)
                else:
                    ba += uint_to_bitarray(resid_num_real[c, r, 0], bits_resid)
                    ba += uint_to_bitarray(resid_num_imag[c, r, 0], bits_resid)
                    # ba += float_to_bitarray(resid[c, r, 0].real)
                    # ba += float_to_bitarray(resid[c, r, 0].imag)
        
        # To file
        with open(filename, 'wb') as fh:
            ba.tofile(fh)

    def decode_file(self, filename, img_shape):

        # Read from file
        ba = bitarray.bitarray()
        with open(filename, 'rb') as fh:
            ba.fromfile(fh)

        def bitarray_to_float(ba_seg):
            return struct.unpack('>f', ba_seg)[0]

        # Get dimensions first
        bits_read = 0
        nr = bitarray_to_uint(ba[bits_read:bits_read+32])
        bits_read += 32
        nc = bitarray_to_uint(ba[bits_read:bits_read+32])
        bits_read += 32
        M = bitarray_to_uint(ba[bits_read:bits_read+32])
        bits_read += 32
        bits_resid = bitarray_to_uint(ba[bits_read:bits_read+32])
        bits_read += 32
        mn_resid = bitarray_to_float(ba[bits_read:bits_read+32])
        bits_read += 32
        mx_resid = bitarray_to_float(ba[bits_read:bits_read+32])
        bits_read += 32

        print(nr, nc, M, bits_resid, mn_resid, mx_resid)
        
        # Quanitzation levels
        levels = np.linspace(mn_resid, mx_resid, 2 ** bits_resid)

        # Make return data
        trj_tc = np.zeros((nr, 1, 2), dtype=np.float32)
        ksp_tc = np.zeros((nc, nr, 1), dtype=np.complex64)
        resid = np.zeros((nc, nr, 1), dtype=np.complex64)
        
        for r in range(nr):

            # Read trajectory
            # if r % M == 0:
            trj_tc[r, 0, 0] = bitarray_to_float(ba[bits_read:bits_read+32])
            bits_read += 32
            trj_tc[r, 0, 1] = bitarray_to_float(ba[bits_read:bits_read+32])
            bits_read += 32

            # Read all coils data
            for c in range(nc):

                if r % M == 0:
                    ksp_tc[c, r, 0] = bitarray_to_float(ba[bits_read:bits_read+32])
                    bits_read += 32
                    ksp_tc[c, r, 0] += bitarray_to_float(ba[bits_read:bits_read+32]) * 1j
                    bits_read += 32
                else:
                    resid[c, r, 0] = levels[bitarray_to_uint(ba[bits_read:bits_read+bits_resid])]
                    bits_read += bits_resid
                    resid[c, r, 0] += levels[bitarray_to_uint(ba[bits_read:bits_read+bits_resid])] * 1j
                    bits_read += bits_resid
                    # resid[c, r, 0] = bitarray_to_float(ba[bits_read:bits_read+32])
                    # bits_read += 32
                    # resid[c, r, 0] += bitarray_to_float(ba[bits_read:bits_read+32]) * 1j
                    # bits_read += 32
        
        # # Reconstruction
        t = np.arange(nr)
        # trj_tc = interp1d(t[::M], trj_tc[::M, :, :], kind='cubic', axis=0, bounds_error=False, fill_value=0)(t)
        ksp_tc = interp1d(t[::M], ksp_tc[:, ::M, :], kind='cubic', axis=1, bounds_error=False, fill_value=0)(t)
        ksp_tc += -resid

        # Get dcf from trajectory
        dcf_tc = pipe_menon_dcf(trj_tc, img_shape)

        return ksp_tc, trj_tc, dcf_tc
    
    def compress(self, comp_type, ksp):
        """ 
        Applies readout compression to k-space data.

        Inputs:
            comp_type - string decribing compression type
            ksp - acquired data tesnsor, complex valued.
                Shape = (d, a, b)
                    d - coil dimension
                    a - same as self.trj.shape[0], readout dimension
                    b - same as self.trj.shape[1], TR/shot/interleave dimension

        This function will compress along the 'a' dimension.
        
        Returns:
            ksp_tc - compressed k-space data
            trj_tc - compressed trajectory
            dcf_tc - compressed density compensation function
        """
        assert comp_type in self.comp_types

        if comp_type == 'Cubic':
            return self.compress_polynomial(ksp)
        elif comp_type == 'TVLPF':
            return self.compress_specgram(ksp)
        elif comp_type == 'Zero Order':
            return self.compress_zero_order_hold(ksp)
        else:
            return self.compress_none(ksp)

    def compress_none(self, ksp):
        """
        Does not compress
        """

        return ksp.copy(), self.trj.copy(), self.dcf.copy()
    
    def compress_zeros(self, ksp, thresh=5e-5):

        inds = np.argwhere(np.max(np.abs(ksp), axis=0) > thresh)
        ksp_tc = ksp[:, inds[:, 0], :]
        trj_tc = self.trj[inds[:, 0], :, :]
        dcf_tc = self.dcf[inds[:, 0], :]
        return ksp_tc, trj_tc, dcf_tc
    
    def compress_zero_order_hold(self, ksp):

        # I don't like retyping self
        dcf = self.dcf
        trj = self.trj
        t_tc = self.t_tc
        npts_tc = len(t_tc)
        t = np.arange(ksp.shape[1])

        # Nearest interpolation
        ksp_tc = np.zeros((ksp.shape[0], npts_tc, ksp.shape[2]), dtype=ksp.dtype)
        trj_tc = np.zeros((npts_tc, trj.shape[1], trj.shape[2]), dtype=trj.dtype)
        dcf_tc = np.zeros((npts_tc, dcf.shape[1]), dtype=dcf.dtype)

        trj_tc = interp1d(t, trj, kind='nearest', axis=0)(t_tc)
        dcf_tc= interp1d(t, dcf, kind='nearest', axis=0)(t_tc)
        ksp_tc = interp1d(t, ksp, kind='nearest', axis=1)(t_tc)

        return ksp_tc, trj_tc, dcf_tc

    def compress_polynomial(self, ksp, interp_kind='cubic', manual_x_axis=None):
        
        # I don't like retyping self
        t_tc = self.t_tc
        trj_tc = self.trj_tc
        dcf_tc = self.dcf_tc
        ind_start = self.ind_start
        npts_tc = len(t_tc)

        # Resample our k-space using interpolator of our choice. Cubic works well enough
        ksp_tc = np.zeros((ksp.shape[0], npts_tc, ksp.shape[2]), dtype=ksp.dtype)
        ksp_tc[:, ind_start:, :] = interp1d(np.arange(ksp.shape[1]), ksp, kind=interp_kind, axis=1)(t_tc[ind_start:])
        ksp_tc[:, :ind_start, :] = ksp[:, :ind_start, :]

        return ksp_tc, trj_tc, dcf_tc
    
    def compress_specgram(self, ksp):
        
        # I don't like retyping self
        trj = self.trj
        t_tc = self.t_tc
        trj_tc = self.trj_tc
        dcf_tc = self.dcf_tc
        ind_start = self.ind_start

        # First compute diff curve
        ni=0
        trj_diffs = np.sqrt(np.sum(np.diff(trj[:, ni, :], axis=0) ** 2, axis=-1))
        trj_diffs = np.append(trj_diffs, trj_diffs[-1])

        trj_diffs_tc = np.sqrt(np.sum(np.diff(trj_tc[:, ni, :], axis=0) ** 2, axis=-1))
        trj_diffs_tc = np.append(trj_diffs_tc, trj_diffs_tc[-1])

        ksp_tc = np.zeros((ksp.shape[0], len(t_tc), ksp.shape[2]), dtype=ksp.dtype)
        def kern(t, T):
            return np.sinc(t / T)
        W = 51
        win = get_window('hamming', W)
        for i in range(ind_start, len(t_tc)):
            ns = np.arange(np.ceil(t_tc[i] - W/2), np.floor(t_tc[i] + W/2) + 1, dtype=int) % trj.shape[0]
            # T = 0.95 / trj_diffs_tc[i]
            T = 0.95 / trj_diffs[int(t_tc[i])]
            k = kern(t_tc[i] - ns, T) * win
            k /= np.sum(k)
            ksp_tc[:, i, :] += np.dot(k, ksp[:, ns, :])

        ksp_tc[:, :ind_start, :] = ksp[:, :ind_start, :]

        return ksp_tc, trj_tc, dcf_tc

    def resample_trajectory(self, interp_type='linear'):

        # I don't like typing self everytime
        trj = self.trj
        dcf = self.dcf
        alpha = self.alpha
        cutoff = self.cutoff
        dkmax = self.dkmax

        # Interleave index
        ni = 0

        # time axis
        t = np.arange(trj.shape[0])

        # Current max radius
        rmax = np.sqrt(np.sum(trj[-1, ni, :] ** 2))

        # Compute starting point
        ind_start = np.argwhere(np.sqrt(np.sum(trj[:, ni, :] ** 2, axis=-1)) > cutoff * rmax).flatten()[0]

        # Redefine terms over the rest of the points
        trj_rest = trj[ind_start:, ni, :]
        dcf_rest = dcf[ind_start:, ni]
        t_rest = t[ind_start:]

        # Parametrized over total kspace distance traveled
        dtrj_dt = np.sqrt(np.sum(np.diff(trj_rest, axis=0) ** 2, axis=-1))
        trj_rest_tot = np.cumsum(dtrj_dt)
        trj_tot_b4 = np.sum(np.sqrt(np.sum(np.diff(trj[:ind_start, ni, :], axis=0) ** 2, axis=-1)))
        trj_rest_tot = np.append(trj_tot_b4, trj_rest_tot + trj_tot_b4)
        radii = np.sqrt(np.sum(trj_rest ** 2, axis=-1))
        radii_interp = interp1d(trj_rest_tot, radii, 'linear') 
        dk_interp = interp1d(trj_rest_tot, np.append(dtrj_dt, dtrj_dt[-1]), 'linear')

        # Now we make our new trajectory
        new_trj = np.array([trj_rest_tot[0]])
        dkmin = np.sqrt(np.sum(np.diff(trj_rest[:10, :], axis=0) ** 2, axis=-1)).min()
        while new_trj[-1] < trj_rest_tot[-1]: 
            val = new_trj[-1]
            dk_old = dk_interp(val)
            delta = (1 - alpha) * dk_old + alpha * dkmax 
            
            # R = radii_interp(val)
            # ratio = R / rmax
            # delta = dkmin * (1 - ratio ** alpha) + dkmax * (ratio ** alpha)
            
            new_trj = np.append(new_trj, val + delta)

        # Old parameterized k-space trajectory
        param = trj_rest_tot

        # New k-space trajectory
        param_tc = new_trj[:-1]

        # Interpolate to new trajectory
        t_rest_tc   = interp1d(param, t_rest,   kind=interp_type, axis=0, assume_sorted=True)(param_tc)
        trj_rest_tc = interp1d(param, trj_rest, kind=interp_type, axis=0, assume_sorted=True)(param_tc)
        dcf_rest_tc = interp1d(param, dcf_rest, kind=interp_type, axis=0, assume_sorted=True)(param_tc)
        npts_tc = len(param_tc) + ind_start

        # Time compressed quanities
        t_tc = np.zeros(npts_tc)
        trj_tc = np.zeros((npts_tc, trj.shape[1], trj.shape[2]), dtype=trj.dtype)
        dcf_tc = np.zeros((npts_tc, dcf.shape[1]), dtype=dcf.dtype)
        trj_tc[:ind_start, ni, :] = trj[:ind_start, ni, :]
        dcf_tc[:ind_start, ni] = dcf[:ind_start, ni]
        t_tc[:ind_start] = t[:ind_start]
        trj_tc[ind_start:, ni, :] = trj_rest_tc
        dcf_tc[ind_start:, ni] = dcf_rest_tc
        t_tc[ind_start:] = t_rest_tc

        return t_tc, trj_tc, dcf_tc, ind_start 
    
    def compress_kmeans(self, ksp, k=1000):
        """
        We can try and compress our K-space trajectory down to a much smaller number of points
        by using this k-means algorithm. We will organize our data into the following matrix:
            |--                                                --|
            |y(kx[0], ky[0]).real, ... , y(kx[M-1], ky[M-1]).real|
        Y = |y(kx[0], ky[0]).imag, ... , y(kx[M-1], ky[M-1]).imag|
            |     kx[0],           ... ,       kx[M-1]           |  
            |     ky[0],           ... ,       ky[M-1]           | 
            |--                                                --|
        Where y(kx, ky) is the complex data frequency domain data at the k-space positions
        kx, ky.

        We then perform K-means on the data matrix Y defined above. 
        """

        # Hate typing self
        trj = self.trj
        dcf = self.dcf

        from sklearn.cluster import KMeans
        from sigpy.mri.dcf import pipe_menon_dcf

        # New matrices
        ksp_tc = np.zeros((ksp.shape[0], k, 1), dtype=ksp.dtype)
        trj_tc = np.zeros((k, 1, trj.shape[-1]), dtype=trj.dtype)
        dcf_tc = np.zeros((k, 1), dtype=dcf.dtype)
        
        # Construct data matrix for each coil
        kx = trj[:, :, 0].flatten()
        ky = trj[:, :, 1].flatten()
        ncoils = ksp.shape[0]
        Y = np.zeros((0, 4))
        for c in range(ncoils):
            ksp_coil = ksp[c, ...].flatten()
            Y = np.vstack((Y, np.array([ksp_coil.real, ksp_coil.imag, kx, ky]).T))

        kmeans = KMeans(n_clusters=k, random_state=0).fit(Y)
        ksp_tc[0, :, 0] = kmeans.cluster_centers_[:, 0] + 1j * kmeans.cluster_centers_[:, 1]
        trj_tc[:, 0, 0] = kmeans.cluster_centers_[:, 2]
        trj_tc[:, 0, 1] = kmeans.cluster_centers_[:, 3]
        # dcf_tc = pipe_menon_dcf(trj_tc, (220, 220)) # HARDCODED WARNING
        for i in range(k):
            dcf_tc[i] = np.sum(kmeans.labels_ == i) / Y.shape[0]

        return ksp_tc, trj_tc, dcf_tc

    
    def compress_interleave(self, ksp):

        # I don't like typing self everytime.
        Rt = self.Rt
        trj = self.trj

        # SVD on K 
        start = time.perf_counter()
        ksp = np.transpose(ksp, (2, 0, 1))
        U, s, Vt = np.linalg.svd(ksp)
        end = time.perf_counter()
        run_time = end - start
        import matplotlib
        matplotlib.use('WebAgg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('Singular Values of Interleave Readout Matrix')
        plt.xlabel('Virtual Interleave Number')
        plt.ylabel('Singular Value Magnitude')
        for i, s_lst in enumerate(s):
            plt.plot(s_lst / s_lst[0])
        plt.legend()
        plt.show()
        print(f'Run Time = {run_time:.3f} sec')
        quit()