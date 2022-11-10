import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Data
data_dir = 'data/matlab_sim_2d/'
# has shape (number of coils, number of samples along trajectory, number of trajectories)
# This is the actualy frequency domain data (complex valued)
ksp = np.load(data_dir + 'ksp.npy') 
# has shape (number of samples along trajectory, number of trajectories, frequency domain spatial index (x y position))
# This tells us where we are in frequency domain
trj = np.load(data_dir + 'trj.npy')
# has shape (number of samples along trajectory, number of trajectories)
# This tells us the sampling density, used for non-cartesian reconstruction algorithms
dcf = np.load(data_dir + 'dcf.npy')
# has shape (number of coils, img_shape[0], img_shape[1])
# This tell us the sensitivty profile for each coil at each spatial position
mps = np.load(data_dir + 'mps.npy')
# This is the sampling rate that was used
dt = 2e-6 # seconds
ncoils, nsamples, nshots = ksp.shape

# Vector quanitization via k-means
# kmeans = KMeans(n_clusters=2 ** 5, random_state=0).fit(X)
for shot in range(nshots):
    for coil in range(ncoils):
        data = ksp[coil, :, shot]
        data_mat = np.array([data.real, data.imag]).T
        absdata = np.abs(data)
        absdata /= absdata.max()
        kmeans = KMeans(n_clusters=2 ** 8, random_state=0).fit(data_mat, sample_weight=absdata ** 2)
        plt.scatter(data.real, data.imag)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='r', marker='x')
        plt.show()
        quit()
        # K-means (unweighted)



