import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

import numpy as np
import time
import sigpy as sp
import struct
import bitarray
import zlib

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

    def __init__(self, trj, dcf, alpha=1, cutoff=0, dkmax=0.95, device=None):
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
        if device is None:
            self.device = sp.get_device(trj)
        else:
            self.device = device
        
        # Resampling
        t_tc, trj_tc, dcf_tc, ind_start = self.resample_trajectory()
        self.tc_weights = np.ones_like(dcf_tc)

        # Save resampled trajectory values
        self.t_tc = t_tc
        self.trj_tc = trj_tc
        self.dcf_tc = dcf_tc
        self.ind_start = ind_start

    def encode_file(self, filename, comp_type, ksp, trj_start=300, trj_skip=40, ksp_skip=20, bits_resid=5, use_lossless=True):

        # ---------------- Compression parameters ---------------- 
        if comp_type == 'Fully Sampled' or comp_type == 'FS Noisy':
            # No compression with this set of parameters
            trj_start = 0
            trj_skip = 1
            ksp_skip = 1
            bits_resid = 0
            use_lossless = False
                
        # ---------------- Compress frequency data and coordinates ----------------
        # Compress kspace with smart decimation first
        ksp_dec, trj_dec, dcf_dec = self.compress_decimate(comp_type, ksp)

        # Compress k-space data with predictive coding
        ksp_comp, real_centers, imag_centers = self.compress_predictive(ksp_dec, trj_dec, ksp_skip, bits_resid)

        # Compress Trajectory
        trj_mag_comp, trj_phase_comp = self.compress_trj(trj_dec, trj_skip, trj_start)

        # ---------------- Save to bitarray ---------------- 
        meta = bitarray.bitarray()
        
        def float_to_bitarray(flt_val):
            ba = bitarray.bitarray()
            flt = struct.pack('>f', flt_val)
            ba.frombytes(flt)
            return ba

        # Meta Data
        nr = ksp_dec.shape[1]
        nc = ksp_dec.shape[0]
        meta += uint_to_bitarray(nr, 32)
        meta += uint_to_bitarray(nc, 32)
        meta += uint_to_bitarray(ksp_skip, 32)
        meta += uint_to_bitarray(trj_skip, 32)
        meta += uint_to_bitarray(trj_start, 32)
        meta += uint_to_bitarray(bits_resid, 32)
        for i in range(2 ** bits_resid):
            meta += float_to_bitarray(real_centers[i])
            meta += float_to_bitarray(imag_centers[i])

        dat = bitarray.bitarray()
        # Encode trajectory
        for r in range(trj_mag_comp.shape[0]):
            dat += float_to_bitarray(trj_mag_comp[r, 0])
            dat += float_to_bitarray(trj_phase_comp[r, 0])

        # Encode DCF
        for r in range(nr):
            dat += float_to_bitarray(dcf_dec[r, 0])

         # Encode all coils data
        for r in range(nr):
            for c in range(nc):

                if r % ksp_skip == 0:
                    dat += float_to_bitarray(ksp_comp[c, r, 0].real)
                    dat += float_to_bitarray(ksp_comp[c, r, 0].imag)
                else:
                    dat += uint_to_bitarray(int(ksp_comp[c, r, 0].real), bits_resid)
                    dat += uint_to_bitarray(int(ksp_comp[c, r, 0].imag), bits_resid)

        # Lossless compression
        if use_lossless:
            meta += uint_to_bitarray(len(dat), 32)
            ret = bitarray.bitarray()
            ret.frombytes(zlib.compress(dat))
            dat = ret
        else:
            meta += uint_to_bitarray(0, 32)

        
        # To file
        with open(filename, 'wb') as fh:
            (meta + dat).tofile(fh)

    def decode_file(self, filename):

        # Read from file
        ba = bitarray.bitarray()
        with open(filename, 'rb') as fh:
            ba.fromfile(fh)

        def bitarray_to_float(ba_seg):
            return struct.unpack('>f', ba_seg)[0]

        # Get meta data first
        bits_read_meta = 0
        nr = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32
        nc = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32
        ksp_skip = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32
        trj_skip = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32
        trj_start = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32
        bits_resid = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32
        real_centers = np.zeros(2 ** bits_resid, dtype=np.float32)
        imag_centers = np.zeros(2 ** bits_resid, dtype=np.float32)
        for i in range(2 ** bits_resid):
            real_centers[i] = bitarray_to_float(ba[bits_read_meta:bits_read_meta+32])
            bits_read_meta += 32
            imag_centers[i] = bitarray_to_float(ba[bits_read_meta:bits_read_meta+32])
            bits_read_meta += 32
        nlossless = bitarray_to_uint(ba[bits_read_meta:bits_read_meta+32])
        bits_read_meta += 32

        # Actual data now
        dat = ba[bits_read_meta:]
        if nlossless != 0:
            ret = bitarray.bitarray()
            ret.frombytes(zlib.decompress(dat))
            dat = ret[:nlossless]
        bits_read = 0

        # Make return data
        trj_mag_comp = np.zeros((nr, 1), dtype=np.float32)
        trj_phase_comp = np.zeros((nr, 1), dtype=np.float32)
        trj_tc = np.zeros((nr, 1, 2), dtype=np.float32)
        dcf_tc = np.zeros((nr, 1), dtype=np.float32)
        ksp_tc = np.zeros((nc, nr, 1), dtype=np.complex64)
        resid = np.zeros((nc, nr, 1), dtype=np.complex64)
        trj_pts = np.append(np.arange(trj_start), np.arange(trj_start, nr, trj_skip))

        # Read trajectory
        for r in trj_pts:
            trj_mag_comp[r, 0] = bitarray_to_float(dat[bits_read:bits_read+32])
            bits_read += 32
            trj_phase_comp[r, 0] = bitarray_to_float(dat[bits_read:bits_read+32])
            bits_read += 32

        # Read DCF
        for r in range(nr):
            dcf_tc[r, 0] = bitarray_to_float(dat[bits_read:bits_read+32])
            bits_read += 32

        # Read all coils data
        for r in range(nr):
            for c in range(nc):

                if r % ksp_skip == 0:
                    ksp_tc[c, r, 0] = bitarray_to_float(dat[bits_read:bits_read+32])
                    bits_read += 32
                    ksp_tc[c, r, 0] += bitarray_to_float(dat[bits_read:bits_read+32]) * 1j
                    bits_read += 32
                else:
                    resid[c, r, 0] = real_centers[bitarray_to_uint(dat[bits_read:bits_read+bits_resid])]
                    bits_read += bits_resid
                    resid[c, r, 0] += imag_centers[bitarray_to_uint(dat[bits_read:bits_read+bits_resid])] * 1j
                    bits_read += bits_resid
        
        # Reconstruction
        t = np.arange(nr)
        trj_mag = interp1d(t[trj_pts], trj_mag_comp[trj_pts, :], kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')(t)
        trj_phase = interp1d(t[trj_pts], trj_phase_comp[trj_pts, :], kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')(t)
        trj_tc[:, :, 0] = trj_mag * np.cos(trj_phase)
        trj_tc[:, :, 1] = trj_mag * np.sin(trj_phase)
        ksp_tc = interp1d(t[::ksp_skip], ksp_tc[:, ::ksp_skip, :], kind='cubic', axis=1, bounds_error=False, fill_value=0)(t)
        ksp_tc += -resid

        return ksp_tc, trj_tc, dcf_tc

    def compress_predictive(self, ksp_dec, trj_dec, ksp_skip, bits_resid):

        if bits_resid == 0 and ksp_skip == 1:
            return ksp_dec, np.array([0]), np.array([0])

        # Downsample by ksp_skip
        t = np.arange(ksp_dec.shape[1])
        
        # Recon and find residuals
        ksp_est = interp1d(t[::ksp_skip], ksp_dec[:, ::ksp_skip, :], kind='cubic', axis=1, bounds_error=False, fill_value=0)(t)
        resid = ksp_est - ksp_dec

        from sklearn.cluster import KMeans

        # Weight center more than edges 
        weights = np.ones(resid.shape)
        assert weights.shape == resid.shape
        # weights[:, ::ksp_skip, :] = 0 # Ignore 0 resid
        weights = weights.reshape(-1, 1)

        real_kmeans = KMeans(n_clusters=2 ** bits_resid, random_state=0).fit(resid.real.reshape((-1, 1)), sample_weight=weights[:, 0])
        imag_kmeans = KMeans(n_clusters=2 ** bits_resid, random_state=0).fit(resid.imag.reshape((-1, 1)))

        resid_real_labels = real_kmeans.labels_.reshape(resid.shape)
        resid_imag_labels = imag_kmeans.labels_.reshape(resid.shape)

        # # Quanitzation
        # mn_resid = min(resid.real.min(), resid.imag.min())
        # mx_resid = max(resid.real.max(), resid.imag.max())
        # levels = np.linspace(mn_resid, mx_resid, 2 ** bits_resid)
        # resid_num_real = np.argmin(np.abs(resid[:, :, :, np.newaxis].real - levels[np.newaxis, np.newaxis, np.newaxis, :]), axis=-1)
        # resid_num_imag = np.argmin(np.abs(resid[:, :, :, np.newaxis].imag - levels[np.newaxis, np.newaxis, np.newaxis, :]), axis=-1)

        # If interpolated point, compress residual difference (a lot)!
        resid_inds = t
        resid_inds = np.delete(resid_inds, t[::ksp_skip])
        ksp_comp = ksp_dec.copy()
        ksp_comp[:, resid_inds, :] = resid_real_labels[:, resid_inds, :] + 1j * resid_imag_labels[:, resid_inds, :]

        return ksp_comp, real_kmeans.cluster_centers_.flatten(), imag_kmeans.cluster_centers_.flatten()

    def compress_trj(self, trj, trj_skip, trj_start):

        # Convert to magnitude phase
        trj_complex = trj[:, :, 0] + 1j * trj[:, :, 1]
        trj_mag = np.abs(trj_complex)
        trj_phase = np.unwrap(np.angle(trj_complex), axis=0)

        # Send every M points and use cubic interpolation to fill in gaps
        pts = np.append(np.arange(trj_start), np.arange(trj_start, trj_mag.shape[0], trj_skip))
        trj_mag_comp = trj_mag[pts, :]
        trj_phase_comp = trj_phase[pts, :]

        return trj_mag_comp, trj_phase_comp
        
    def compress_decimate(self, comp_type, ksp):
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
        # assert comp_type in self.comp_types

        if 'Cubic' in comp_type:
            return self.compress_polynomial(ksp)
        elif 'TVLPF' in comp_type:
            return self.compress_specgram(ksp)
        elif 'Zero Order' in comp_type:
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

        interp = 'nearest'
        trj_tc = interp1d(t, trj, kind=interp, axis=0)(t_tc)
        dcf_tc= interp1d(t, dcf, kind=interp, axis=0)(t_tc)
        ksp_tc = interp1d(t, ksp, kind=interp, axis=1)(t_tc)

        return ksp_tc, trj_tc, dcf_tc

    def compress_polynomial(self, ksp, interp_kind='cubic'):
        
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
            self.tc_weights[i, :] = np.sum(k ** 2)
        # plt.plot(tc_weights)
        ksp_tc[:, :ind_start, :] = ksp[:, :ind_start, :]
        exp = 1
        print(f'EXP = {exp}')
        return ksp_tc, trj_tc, dcf_tc / (self.tc_weights ** exp)

    def resample_trajectory(self):

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
        dk_interp = interp1d(trj_rest_tot, np.append(dtrj_dt, dtrj_dt[-1]), 'linear')

        # Now we make our new trajectory
        new_trj = np.array([trj_rest_tot[0]])
        # dkmax = np.max(dtrj_dt)
        while new_trj[-1] < trj_rest_tot[-1]: 
            val = new_trj[-1]
            dk_old = dk_interp(val)
            delta = (1 - alpha) * dk_old + alpha * dkmax 
            new_trj = np.append(new_trj, val + delta)

        # Old parameterized k-space trajectory
        param = trj_rest_tot

        # New k-space trajectory
        param_tc = new_trj[:-1]

        # Interpolate to new trajectory
        interp_type = 'linear'
        t_rest_tc   = interp1d(param, t_rest,   kind=interp_type, axis=0, assume_sorted=True)(param_tc)
        npts_tc = len(param_tc) + ind_start

        # Time compressed quanities
        t_tc = np.zeros(npts_tc)
        t_tc[:ind_start] = t[:ind_start]
        t_tc[ind_start:] = t_rest_tc
        t_tc = np.unique(t_tc)


        trj_tc = interp1d(t, trj, kind=interp_type, axis=0, assume_sorted=True)(t_tc)
        dcf_tc = interp1d(t, dcf, kind=interp_type, axis=0, assume_sorted=True)(t_tc)

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