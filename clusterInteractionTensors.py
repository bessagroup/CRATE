#
# Cluster Interaction Tensors Computation Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Generate efficient iterators
import itertools as it
# Tensorial operations
import tensorOperations as top
#
#                                                        Compute cluster interaction tensors
# ==========================================================================================
# Compute the fourth-order cluster interaction tensors between the different material phases
# and store them in matricial form as described below.
#
# Lets assume that the heterogeneous material has 3 phases, namely '1', '2' and '3'. Given
# that the clustering is performed independently on each material phase, the cluster
# interaction tensors must be computed for every pair resulting from the cross combination
# of the different material phases, i.e. '11', '12', '13', '21', '22', 'BC', 'CA', 'CB',
# 'CC'. Taking for example the pair '12' and assuming that the material phase '1' contains
# the clusters '0', '1' and '2', and the material phase '2' the clusters '3' and '4', the
# clustering interaction tensors of the pair '12' are stored as
#
#   dictionary['12'] = {'03':cit_mf , '04':cit_mf, '13':cit_mf, '14:cit_mf', ...},
#
#                      where 'ij':cit_mf is the matricial form of the cluster interaction
#                      tensor between the clusters i and j
#
def computeClusterInteractionTensors():
    # Initialize cluster interaction tensors dictionary
    cit = dict()
    # Compute Green operator material independent terms
    Gop_1_DFT_vox, Gop_2_DFT_vox, Gop_0_freq_DFT_vox = \
                         GreenOperatorMatIndTerms(n_dim,comp_order,n_voxels_dims,freqs_dims)
    # Loop over material phases
    for mat_phase_B in phase_clusters.keys():
        for mat_phase_A in phase_clusters.keys():
            # Initialize material phases pair cluster interaction tensors dictionary
            cit[phaseA+phaseB] = dict()
            # Loop over material phase B clusters
            for clusterJ in phase_clusters[mat_phase_B]:



#
#                                                                    Complementary functions
# ==========================================================================================
# Perform the frequency discretization by setting the spatial discrete frequencies (rad/m)
# for each dimension
def setDiscreteFrequencies(n_dim,rve_dims,n_voxels_dims):
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*math.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    # Return
    return freq_dims
# ------------------------------------------------------------------------------------------
# Computes the material independent terms of the Green operator (in the frequency domain)
# for every pixel/voxel and store them as follows:
#
# A. 2D problem (plane strain):
#
#   Gop_1_DFT_vox[comp]      = array(d1,d2),
#   Gop_2_DFT_vox[comp]      = array(d1,d2),
#   Gop_0_DFT_freq_vox[comp] = array(d1,d2),
#
#                        where | di is the number of pixels in dimension i
#                              | comp is the component that would be stored in matricial
#                              |      form ('1111', '2211', '1211', '1122', ...)
#
# B. 3D problem:
#
#   Gop_1_DFT_vox[comp]      = array(d1,d2,d3),
#   Gop_2_DFT_vox[comp]      = array(d1,d2,d3),
#   Gop_0_DFT_freq_vox[comp] = array(d1,d2,d3),
#
#                        where | di is the number of pixels in dimension i
#                              | comp is the component that would be stored in matricial
#                              |     form ('1111', '2211', '3311', '1211', ...)
#
# Note: The material independent terms of the Green operator, as well as the zero-frequency
#       term, are conveniently computed to perform an efficient update of the Green operator
#       if the reference material elastic properties are updated (e.g. self-consistent
#       scheme)
#
def GreenOperatorMatIndTerms(n_dim,comp_order,n_voxels_dims,freqs_dims):
    # Set Green operator matricial form components
    comps = list(it.product(comp_order,comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x)-1 for x in list(comps[i][0]+comps[i][1])])
        mf_indexes.append([x for x in \
                             [comp_order.index(comps[i][0]),comp_order.index(comps[i][1])]])
    # Initialize Green operator material independent terms
    Gop_1_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    Gop_2_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    Gop_0_freq_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get matrix index
        mf_idx = mf_indexes[i]
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get discrete frequency index
            freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) for \
                                                                         x in range(n_dim)])
            # Set Green operator zero-frequency term to unit at zero-frequency. Skip the
            # zero-frequency computation for the remaining Green operator terms
            if freq_idx == n_dim*(0,):
                Gop_0_freq_vox[comp][freq_idx] = 1.0
                continue
            # Compute frequency vector norm
            freq_norm = np.linalg.norm(freq_coord)
            # Compute first material independent term of Green operator
            Gop_1_DFT_vox[comp][freq_idx] = (1.0/freq_norm**2)*(
                   top.Dd(fo_idx[0],fo_idx[2])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[3]] +
                   top.Dd(fo_idx[0],fo_idx[3])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[2]] +
                   top.Dd(fo_idx[1],fo_idx[3])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[2]] +
                   top.Dd(fo_idx[1],fo_idx[2])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[3]])
            # Compute second material independent term of Green operator
            Gop_2_DFT_vox[comp][freq_idx] = -(1.0/freq_norm**4)*\
                                               (freq_coord[fo_idx[0]]*freq_coord[fo_idx[1]]*
                                               freq_coord[fo_idx[2]]*freq_coord[fo_idx[3]])
    # Return
    return [Gop_1_DFT_vox,Gop_2_DFT_vox,Gop_0_freq_DFT_vox]
# ------------------------------------------------------------------------------------------
# Set the discrete characteristic function associated to a given material cluster, which in
# the spatial domain is equal to 1 for the material cluster domain points and 0 otherwise.
# Also set the Discrete Fourier Transform (DFT) of the discrete characteristic function
def getClusterFilter(cluster,voxels_clusters):
    # Check if valid cluster
    if not isinstance(cluster,int):
        print('error: cluster must be integer')
    elif cluster not in voxels_clusters:
        print('error: unexistent cluster')
    # Build cluster filter (spatial domain)
    cluster_filter = voxels_clusters == cluster
    # Perform Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
    cluster_filter_DFT = np.fft.fftn(cluster_filter)
    # Return
    return [cluster_filter,cluster_filter_DFT]
# ------------------------------------------------------------------------------------------
# Perform the discrete convolution required to compute the cluster interaction tensor
# between a material cluster I and a material cluster J. The convolution is performed in the
# frequency domain between the material cluster J characteristic function (frequency domain)
# and a each Green operator material independent terms (frequency domain). Return the
# convolution in the spatial domain by performing an Inverse Discrete Fourier Transform
# (IDFT)
def discreteCITConvolutionJ(cluster_filter_DFT,Gop_1_DFT_vox,Gop_2_DFT_vox,\
                                                                        Gop_0_freq_DFT_vox):
    # Initialize discrete convolution (frequency domain)
    Gop_1_filt_DFT_vox = copy.deepcopy(Gop_1_DFT_vox)
    Gop_2_filt_DFT_vox = copy.deepcopy(Gop_2_DFT_vox)
    Gop_0_freq_filt_DFT_vox = copy.deepcopy(Gop_0_freq_DFT_vox)
    # Loop over Green operator components
    for i in range(len(comp_order)):
        compi = comp_order[i]
        for j in range(len(comp_order)):
            compj = comp_order[j]
            # Perform discrete convolution in the frequency domain
            Gop_1_filt_DFT_vox[compi+compj] = \
                             np.multiply(cluster_filter_DFT,Gop_1_filt_DFT_vox[compi+compj])
            Gop_2_filt_DFT_vox[compi+compj] = \
                             np.multiply(cluster_filter_DFT,Gop_2_filt_DFT_vox[compi+compj])
            Gop_0_freq_filt_DFT_vox[compi+compj] = \
                        np.multiply(cluster_filter_DFT,Gop_0_freq_filt_DFT_vox[compi+compj])
            # Perform an Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            Gop_1_filt_vox[compi+compj] = \
                                      np.real(np.fft.ifftn(Gop_1_filt_DFT_vox[compi+compj]))
            Gop_2_filt_vox[compi+compj] = \
                                      np.real(np.fft.ifftn(Gop_2_filt_DFT_vox[compi+compj]))
            Gop_0_freq_filt_vox[compi+compj] = \
                                 np.real(np.fft.ifftn(Gop_0_freq_filt_DFT_vox[compi+compj]))
    # Return
    return [Gop_1_filt_vox,Gop_2_filt_vox,Gop_0_freq_filt_vox]
# ------------------------------------------------------------------------------------------
# Perform the discrete integral over the spatial domain of material cluster I required to
# compute the cluster interaction tensor between the material cluster I and a material
# cluster J. Store the resulting fourth-order tensor in matricial form
def discreteCITIntegralI(comp_order,cluster_filter,GOP_1_filt_vox,GOP_2_filt_vox,\
                                                                       GOP_0_freq_filt_vox):
    # Set matricial form components
    comps = list(it.product(comp_order,comp_order))
    # Initialize discrete integral
    cit_1_integral_mf = np.zeros((len(comp_order),len(comp_order)))
    cit_2_integral_mf = np.zeros((len(comp_order),len(comp_order)))
    cit_0_freq_integral_mf = np.zeros((len(comp_order),len(comp_order)))
    # Loop over matricial form components
    for i in range(len(comp_order)):
        compi = comp_order[i]
        for j in range(len(comp_order)):
            compj = comp_order[j]
            # Perform discrete integral over the spatial domain of material cluster I
            cit_1_integral_mf[i,j] = \
                             top.kelvinFactor(i,comp_order)*top.kelvinFactor(j,comp_order)*\
                             np.sum(np.multiply(cluster_filter,GOP_1_filt_vox[compi+compj])
            cit_2_integral_mf[i,j] = \
                             top.kelvinFactor(i,comp_order)*top.kelvinFactor(j,comp_order)*\
                             np.sum(np.multiply(cluster_filter,GOP_2_filt_vox[compi+compj])
            cit_0_freq_integral_mf[i,j] = \
                             top.kelvinFactor(i,comp_order)*top.kelvinFactor(j,comp_order)*\
                        np.sum(np.multiply(cluster_filter,GOP_0_freq_filt_vox[compi+compj]))
    # Return
    return [cit_1_integral_mf,cit_2_integral_mf,cit_0_freq_integral_mf]
