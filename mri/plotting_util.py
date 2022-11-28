# Import libraries
import numpy as np

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

# Plot fonts
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 17}
matplotlib.rc('font', **font)

# Set threads
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

class plotting_util:

    def __init__(self, recons, tc_params, snr_func=None, figsize=(14,7), dt=1):
        """
        Inputs:
            recons - a dictionary of reconstructed images. Should look like:
                - recons['Fully Sampled'].shape = (N, img_shape[0], img_shape[1])
                - recons.keys() may be = ['Fully Sampled', 'FS Noisy', 'TVLPF', ...]
                - N is the number of images we have
            tc_params - a dictionary with:
                - tc_params['Fully Sampled].shape = (N, 4)
                4 --> alpha, cutoff, dkmax, Rt
        """
        self.recons = recons
        self.tc_params = tc_params
        self.dt = dt
        self.figsize = figsize
        if snr_func is None:
            def snr_func(img):
                noise_std = np.std(img[89:128, 3:18].flatten())
                sig_strn = np.mean(np.abs(img[82:127, 82:127].flatten()))
                return sig_strn / noise_std
            self.snr_func = snr_func
        else:
            self.snr_func = snr_func

        assert 'Fully Sampled' in recons

    def trajectory_figure(self, trj_fs, trj_tc):

        # Setup figure
        fig_trj, axs_trj = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        M = 5 # Skip M samples

        # Show trajectories
        axs_trj.scatter(trj_tc[::M, 0, 0], trj_tc[::M, 0, 1], color='red', marker='x', alpha=0.5, label='Resampled')
        axs_trj.plot(trj_fs[:, 0, 0], trj_fs[:, 0, 1], color='black', marker='.', alpha = 0.5, label='Fully Sampled')
        axs_trj.axis('off')
        axs_trj.legend()
    
    def delta_kspace_figure(self, trj_fs, trj_tc):

        # Setup figure
        fig_diff, axs_trj_diff = plt.subplots(1, 1, figsize=self.figsize)
        axs_trj_diff.set_ylabel(r'$||\Delta_k||$')

        
        # Compute kspace deltas
        trj_tc_diffs = np.sqrt(np.sum(np.diff(trj_tc, axis=0) ** 2, axis=2))
        trj_fs_diffs = np.sqrt(np.sum(np.diff(trj_fs, axis=0) ** 2, axis=2))
        t = np.arange(trj_tc.shape[0] - 1) * self.dt

        # One interleave is fine
        ni = 0

        axs_trj_diff.axhline(1, color='black', linestyle='--', label=r'$\frac{1}{FOV}$')

        axs_trj_diff.plot(t * 1e3, trj_fs_diffs[:, ni], label='Fully Sampled', alpha=1, color='r')
        axs_trj_diff.plot(t * 1e3, trj_tc_diffs[:, ni], label='Resampled', alpha=1, color='g')
            
        axs_trj_diff.legend()    
        axs_trj_diff.set_xlabel('Time [ms]')

    def recon_figure(self):

        assert len(self.recons['Fully Sampled'].shape) == 2
        
        # Setup figure
        fig_recon, axs_recon = plt.subplots(nrows=2, 
                                ncols=len(self.recons), 
                                gridspec_kw={'wspace':0, 'hspace':0},
                                figsize=self.figsize,
                                squeeze=True)

        # Plot all
        for i, comp_type in enumerate(self.recons.keys()):
            
            # Get time compressed recon
            recon_tc = self.recons[comp_type]

            # Compute SNR
            SNR = self.snr_func(recon_tc)

            # Max/min from fully sampled
            if comp_type == 'Fully Sampled':
                recon_fs = recon_tc
                vmin = np.abs(recon_fs).min()
                # hist, vals = np.histogram(np.abs(recon_fs).flatten(), bins=50)
                # mxs = argrelextrema(hist, np.greater)[0][-1]
                # vmax = vals[mxs]
                vmax = np.abs(recon_fs).max()

            # Normalize
            recon_tc = self.normalize(recon_tc, recon_fs)

            # Recon figure
            if len(self.recons) == 1:
                ax_recon = axs_recon[0]
                ax_recon_diff = axs_recon[1]
            else:
                ax_recon = axs_recon[0, i]
                ax_recon_diff = axs_recon[1, i]
            ax_recon.set_title(comp_type)
            ax_recon.imshow(np.abs(recon_tc), cmap='gray', vmin=vmin, vmax=vmax)
            # ax_recon.set_ylim(70, 126)
            # ax_recon.set_xlim(25, 56)
            ax_recon.axis('off')
            ax_recon_diff.axis('off')
            err = np.abs(np.abs(recon_fs) - np.abs(recon_tc))
            ax_recon_diff.imshow(err, cmap='gray', vmin=0, vmax=vmax/5)

            # Report MSE/SNR/RT
            MSE = np.mean(np.abs(err) ** 2, axis=(0, 1))
            print(f'{comp_type:<20}SNR = {SNR:.2f}', end=', ')
            print(f'MSE = {MSE:.3e}', end=', ')
            print(f'Rt = {self.tc_params[comp_type][-1]}')

    def tc_animation(self, file_gif):

        assert len(self.recons['Fully Sampled'].shape) == 3
        N = len(self.recons['Fully Sampled'])
        fig, axs = plt.subplots(2, len(self.recons.keys()), 
                                figsize=(20, 9),
                                gridspec_kw={'wspace':0, 'hspace':0},
                                squeeze=True)
        frames = [] 
        inds = np.arange(N)
        inds = np.append(inds, np.flip(inds))
        for k in inds:
            frame = []
            for i, comp_type in enumerate(self.recons.keys()):

                recon_tc = self.recons[comp_type][k]

                # Max/min from fully sampled
                if comp_type == 'Fully Sampled':
                    recon_fs = recon_tc
                    vmin = np.abs(recon_fs).min()
                    vmax = np.abs(recon_fs).max()

                # Normalize
                recon_tc = self.normalize(recon_tc, recon_fs)

                # Recon figure
                ax_recon = axs[0, i]
                ax_recon_diff = axs[1, i]
                ax_recon.set_title(comp_type)
                frame0 = ax_recon.imshow(np.abs(recon_tc), cmap='gray', vmin=vmin, vmax=vmax)
                ax_recon.axis('off')
                ax_recon_diff.axis('off')
                err = np.abs(np.abs(recon_fs) - np.abs(recon_tc))
                frame1 = ax_recon_diff.imshow(err, cmap='gray', vmin=0, vmax=vmax/5)
            
                if comp_type == 'Fully Sampled':
                    Rt = self.tc_params['Cubic'][k, -1]
                    t = ax_recon_diff.annotate(r'$R_t$' + f' = {Rt:.1f}',(30, 120), fontsize=30, backgroundcolor='w')
                    frame += [frame0, frame1, t]
                else:
                    frame += [frame0, frame1]
            frames.append(frame)
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                        repeat_delay=0)
        ani.save(file_gif + '.gif')

    def tc_plot_figure(self):

        assert len(self.recons['Fully Sampled'].shape) == 3
        N = len(self.recons['Fully Sampled'])

        # All groups of dkmax
        dkmaxs = self.tc_params['Cubic'][:, 2]
        N = np.argwhere(np.diff(dkmaxs) != 0).flatten()[0] + 1
        Rts = self.tc_params['Cubic'][::N, -1]
        ngroups = len(dkmaxs) // N
        plt.figure(figsize=(14,7))
        for comp_type in self.recons.keys():
            mse_means = []
            mse_stds = []
            for r in range(ngroups):
                mses = []
                for k in range(r*N, (r+1)*N):
                    
                    # Get recon
                    recon_tc = self.recons[comp_type][k]

                    # Max/min from fully sampled
                    if comp_type == 'Fully Sampled':
                        recon_fs = recon_tc
                        SNR = self.snr_func(recon_fs)

                    # Normalize
                    recon_tc = self.normalize(recon_tc, recon_fs)

                    # MSE
                    err = np.abs(np.abs(recon_fs) - np.abs(recon_tc))
                    MSE = np.mean(np.abs(err) ** 2, axis=(0, 1))
                    mses.append(MSE)

                # Mean, std
                mse_means.append(np.mean(mses))
                mse_stds.append(np.std(mses))

            if comp_type != 'Fully Sampled':
                plt.errorbar(Rts, mse_means, mse_stds, label=comp_type)
        plt.xlabel('Rt')
        plt.ylabel('MSE')
        plt.legend()

    def gfactor_error_animation(self, file_gif):
        assert 'FS Noisy' in self.recons
        assert len(self.recons['FS Noisy'].shape) == 3

         # Setup figures
        fig_error, axs_error = plt.subplots(nrows=2, 
                                ncols=len(self.recons), 
                                gridspec_kw={'wspace':0, 'hspace':0},
                                figsize=self.figsize,
                                squeeze=True)
        fig_error.suptitle('Error Bias')
        frames_error = []
        fig_gfactor, axs_gfactor = plt.subplots(nrows=1, 
                                ncols=len(self.recons) - 1, 
                                gridspec_kw={'wspace':0, 'hspace':0},
                                figsize=self.figsize,
                                squeeze=True)
        fig_gfactor.suptitle('1/g Factor')
        frames_gfactor = [] 

        # All groups of dkmax
        dkmaxs = self.tc_params['Cubic'][:, 2]
        N = np.argwhere(np.diff(dkmaxs) != 0).flatten()[0] + 1
        ngroups = len(dkmaxs) // N
        iters = np.arange(ngroups)
        iters = np.append(iters, np.flip(iters))
        for r in iters:
        
            # First we get the fully sampled SNR
            std_fs = np.std(self.recons['FS Noisy'][r*N:(r+1)*N, ...], axis=0)        

            # Get time compression
            Rt = self.tc_params['Cubic'][r*N, -1]
                        
            # Go through all compressors
            i = 0
            frame_gfactor = []
            frame_error = []
            for j, comp_type in enumerate(self.recons.keys()):

                # Get time compression
                Rt = self.tc_params['Cubic'][N*r, -1]

                # G-Factor here
                if comp_type != 'Fully Sampled':
                    # Calculate G-factor
                    acceleration = self.tc_params[comp_type][0, 2]
                    std_tc = np.std(self.recons[comp_type][r*N:(r+1)*N], axis=0)
                    g_factor = std_tc / std_fs / np.sqrt(acceleration)

                    # Plot G-factor
                    axs_gfactor[i].set_title(comp_type)
                    im_gfactor = axs_gfactor[i].imshow(1 / g_factor, vmin=0.5, vmax=1, cmap='jet_r')
                    axs_gfactor[i].axis('off')
                    frame_gfactor += [im_gfactor]
                    i += 1

                # Calculate means 
                mean_img = np.mean(self.recons[comp_type][r*N:(r+1)*N], axis=0)
                ref = np.mean(self.recons['Fully Sampled'][r*N:(r+1)*N], axis=0)
                mean_img = self.normalize(mean_img, ref) # Normalize

                # Min and Max
                if comp_type == 'Fully Sampled':
                    vmin = np.abs(ref).min()
                    vmax = np.abs(ref).max()
                
                err = np.abs(np.abs(mean_img) - np.abs(ref))

                # Plot mean and error
                axs_error[0, j].set_title(comp_type)
                axs_error[0, j].axis('off')
                axs_error[1, j].axis('off')
                im_err_0 = axs_error[0, j].imshow(np.abs(mean_img), vmin=vmin, vmax=vmax, cmap='gray')
                im_err_1 = axs_error[1, j].imshow(err, vmin=vmin, vmax=vmax/10, cmap='gray')
                frame_error += [im_err_0, im_err_1]

            # Add text, then add frames
            rt_text = fig_gfactor.text(0.475, 0.25, f'Rt = {Rt:.2f}', fontsize=25)
            frame_gfactor += [rt_text]
            rt_text = fig_error.text(0.475, 0.10, f'Rt = {Rt:.2f}', fontsize=25)
            frame_error += [rt_text]
            frames_gfactor.append(frame_gfactor)
            frames_error.append(frame_error)
        
        # Place colorbar just to the right
        axr = axs_gfactor[-1]
        cax = fig_gfactor.add_axes([axr.get_position().x1+0.01,axr.get_position().y0,0.02,axr.get_position().height])
        fig_gfactor.colorbar(im_gfactor, cax=cax)

        # Animate and save
        ani = animation.ArtistAnimation(fig_gfactor, frames_gfactor, interval=200, blit=True,
                                        repeat_delay=0)
        ani.save(file_gif + '_gfactor.gif')
        ani = animation.ArtistAnimation(fig_error, frames_error, interval=200, blit=True,
                                        repeat_delay=0)
        ani.save(file_gif + '_error.gif')

    def gfactor_error_figure(self):

        assert 'FS Noisy' in self.recons
        assert len(self.recons['FS Noisy'].shape) == 3

        # Find length and dkmax desired
        N = len(self.recons['FS Noisy']) * 0 + 20
        
        # Setup figures
        fig_error, axs_error = plt.subplots(nrows=2, 
                                ncols=len(self.recons), 
                                gridspec_kw={'wspace':0, 'hspace':0},
                                figsize=self.figsize,
                                squeeze=True)
        fig_gfactor, axs_gfactor = plt.subplots(nrows=1, 
                                ncols=len(self.recons) - 1, 
                                gridspec_kw={'wspace':0, 'hspace':0},
                                figsize=self.figsize,
                                squeeze=True)
        

        # First we get the fully sampled SNR
        std_fs = np.std(self.recons['FS Noisy'], axis=0)
        
        # Go through all compressors
        i = 0
        for j, comp_type in enumerate(self.recons.keys()):

            # Get time compression
            Rt = self.tc_params['Cubic'][:N, -1]

            # G-Factor here
            if comp_type != 'Fully Sampled':
                # Calculate G-factor
                acceleration = self.tc_params[comp_type][0, 2]
                std_tc = np.std(self.recons[comp_type][:N], axis=0)
                g_factor = std_tc / std_fs / np.sqrt(acceleration)

                # Plot G-factor
                axs_gfactor[i].set_title(comp_type)
                im = axs_gfactor[i].imshow(1 / g_factor, vmin=0.5, vmax=1, cmap='jet_r')
                axs_gfactor[i].axis('off')
                i += 1

            # Calculate means 
            mean_img = np.mean(self.recons[comp_type][:N], axis=0)
            ref = np.mean(self.recons['Fully Sampled'][:N], axis=0)
            mean_img = self.normalize(mean_img, ref) # Normalize

            if comp_type == 'Fully Sampled':
                vmin = np.abs(ref).min()
                vmax = np.abs(ref).max()
            
            err = np.abs(np.abs(ref) - np.abs(mean_img))

            # Plot mean and error
            axs_error[0, j].set_title(comp_type)
            axs_error[0, j].imshow(np.abs(mean_img), vmin=vmin, vmax=vmax, cmap='gray')
            axs_error[1, j].imshow(err, vmin=vmin, vmax=vmax/5, cmap='gray')
            axs_error[0, j].axis('off')
            axs_error[1, j].axis('off')

        fig_gfactor.suptitle(f'1/g (Rt = {Rt[0]:.2f})')
        fig_error.suptitle('Error Bias')
        axr = axs_gfactor[-1]
        cax = fig_gfactor.add_axes([axr.get_position().x1+0.01,axr.get_position().y0,0.02,axr.get_position().height])
        fig_gfactor.colorbar(im, cax=cax)

    def normalize(self, shifted, target):
        col1 = shifted.flatten()
        col2 = col1 * 0 + 1
        A = np.array([col1, col2]).T
        y = target.flatten()
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return a * shifted + b
