#
# Cluster Interaction Tensors Computation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Methods associated with the computation of the CRVE cluster interaction tensors and their
# assembly into the global cluster interaction matrix.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
import numpy.matlib
# Inspect file name and line
import inspect
# Shallow and deep copy operations
import copy
# Generate efficient iterators
import itertools as it
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                                       Discrete frequencies
# ==========================================================================================
def setdiscretefreq(n_dim, rve_dims, n_voxels_dims):
    '''Perform frequency discretization of the spatial domain.

    Perform frequency discretization by setting the spatial discrete frequencies (rad/m)
    for each dimension.

    Parameters
    ----------
    n_dim : int
        Problem dimension.
    rve_dims : list
        RVE size in each dimension.
    n_voxels_dims : list
        Number of voxels in each dimension of the regular grid (spatial discretization of
        the RVE).

    Returns
    -------
    freq_dims : list
        List containing the sample frequencies (ndarray of shape (n_voxels_dim,)) for each
        dimension.
    '''
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i], sampling_period))
    # Return
    return freqs_dims
#
#                                                                             Green operator
# ==========================================================================================
# Computes the material independent terms of the Green operator (in the frequency domain)
# for every pixel/voxel and store them as follows:
#
# A. 2D problem (plane strain):
#
#   gop_1_dft_vox[comp]      = array(d1,d2),
#   gop_2_dft_vox[comp]      = array(d1,d2),
#   gop_0_dft_freq_vox[comp] = array(d1,d2),
#
#                        where | di is the number of pixels in dimension i
#                              | comp is the component that would be stored in matricial
#                              |      form ('1111', '2211', '1211', '1122', ...)
#
# B. 3D problem:
#
#   gop_1_dft_vox[comp]      = array(d1,d2,d3),
#   gop_2_dft_vox[comp]      = array(d1,d2,d3),
#   gop_0_dft_freq_vox[comp] = array(d1,d2,d3),
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
def gop_matindterms(n_dim, rve_dims, comp_order, n_voxels_dims):
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = setdiscretefreq(n_dim, rve_dims, n_voxels_dims)
    # Set Green operator matricial form components
    comps = list(it.product(comp_order, comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
        mf_indexes.append([x for x in \
                           [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
    # Set optimized variables
    var1 = [*np.meshgrid(*freqs_dims, indexing = 'ij')]
    var2 = dict()
    for fo_idx in fo_indexes:
        if str(fo_idx[1]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[1]], var1[fo_idx[3]])
        if str(fo_idx[1]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[1]], var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[0]], var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[0]], var1[fo_idx[3]])
        if ''.join([str(x) for x in fo_idx]) not in var2.keys():
            var2[''.join([str(x) for x in fo_idx])] = \
                np.multiply(np.multiply(var1[fo_idx[0]], var1[fo_idx[1]]),
                            np.multiply(var1[fo_idx[2]], var1[fo_idx[3]]))
    if n_dim == 2:
        var3 = np.sqrt(np.add(np.square(var1[0]), np.square(var1[1])))
    else:
        var3 = np.sqrt(np.add(np.add(np.square(var1[0]), np.square(var1[1])),
                              np.square(var1[2])))
    # Initialize Green operator material independent terms
    gop_1_dft_vox = {''.join([str(x+1) for x in idx]): \
                     np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    gop_2_dft_vox = {''.join([str(x+1) for x in idx]): \
                     np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    gop_0_freq_dft_vox = {''.join([str(x+1) for x in idx]): \
                          np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Set optimized variables
        var4 = [fo_idx[0] == fo_idx[2], fo_idx[0] == fo_idx[3],
                fo_idx[1] == fo_idx[3], fo_idx[1] == fo_idx[2]]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]), str(fo_idx[1]) + str(fo_idx[2]),
                str(fo_idx[0]) + str(fo_idx[2]), str(fo_idx[0]) + str(fo_idx[3])]
        # Compute first material independent term of Green operator
        first_term = np.zeros(tuple(n_voxels_dims))
        for j in range(len(var4)):
            if var4[j]:
                first_term = np.add(first_term, var2[var5[j]])
        first_term = np.divide(first_term, np.square(var3), where = abs(var3) > 1e-10)
        gop_1_dft_vox[comp] = copy.copy(first_term)
        # Compute second material independent term of Green operator
        gop_2_dft_vox[comp] = -1.0*np.divide(var2[''.join([str(x) for x in fo_idx])],
            np.square(np.square(var3)), where = abs(var3) > 1e-10)
        # Compute Green operator zero-frequency term
        gop_0_freq_dft_vox[comp][tuple(n_dim*(0,))] = 1.0
    # Return
    return [gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox]
#
#                                                            Cluster characteristic function
# ==========================================================================================
# Set the discrete characteristic function associated to a given material cluster, which in
# the spatial domain is equal to 1 for the material cluster domain points and 0 otherwise.
# Also set the Discrete Fourier Transform (DFT) of the discrete characteristic function
def clusterfilter(cluster, voxels_clusters):
    # Check if valid cluster
    if not isinstance(cluster, int) and not isinstance(cluster, np.integer):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00040', location.filename, location.lineno + 1)
    elif cluster not in voxels_clusters:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00041', location.filename, location.lineno + 1)
    # Build cluster filter (spatial domain)
    cluster_filter = voxels_clusters == cluster
    # Perform Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
    cluster_filter_dft = np.fft.fftn(cluster_filter)
    # Return
    return [cluster_filter, cluster_filter_dft]
#
#                                              Cluster - Green operator discrete convolution
# ==========================================================================================
# Perform the discrete convolution required to compute the cluster interaction tensor
# between a material cluster I and a material cluster J. The convolution is performed in the
# frequency domain between the material cluster J characteristic function (frequency domain)
# and each of Green operator material independent terms (frequency domain). Return the
# convolution in the spatial domain by performing an Inverse Discrete Fourier Transform
# (IDFT)
def clstgopconvolution(comp_order, rve_dims, n_voxels_dims, cluster_filter_dft,
                       gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox):
    # Initialize discrete convolution (spatial and frequency domain)
    gop_1_filt_dft_vox = copy.deepcopy(gop_1_dft_vox)
    gop_2_filt_dft_vox = copy.deepcopy(gop_2_dft_vox)
    gop_0_freq_filt_dft_vox = copy.deepcopy(gop_0_freq_dft_vox)
    gop_1_filt_vox = copy.deepcopy(gop_1_dft_vox)
    gop_2_filt_vox = copy.deepcopy(gop_1_dft_vox)
    gop_0_freq_filt_vox = copy.deepcopy(gop_1_dft_vox)
    # Compute RVE volume and total number of voxels
    rve_vol = np.prod(rve_dims)
    n_voxels = np.prod(n_voxels_dims)
    # Loop over Green operator components
    for i in range(len(comp_order)):
        compi = comp_order[i]
        for j in range(len(comp_order)):
            compj = comp_order[j]
            # Perform discrete convolution in the frequency domain
            gop_1_filt_dft_vox[compi + compj] = \
                np.multiply((rve_vol/n_voxels), np.multiply(cluster_filter_dft,
                    gop_1_filt_dft_vox[compi + compj]))
            gop_2_filt_dft_vox[compi + compj] = \
                np.multiply((rve_vol/n_voxels), np.multiply(cluster_filter_dft,
                    gop_2_filt_dft_vox[compi + compj]))
            gop_0_freq_filt_dft_vox[compi + compj] = \
                np.multiply((rve_vol/n_voxels),np.multiply(cluster_filter_dft,
                    gop_0_freq_filt_dft_vox[compi + compj]))
            # Perform an Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            gop_1_filt_vox[compi + compj] = \
                np.real(np.fft.ifftn(gop_1_filt_dft_vox[compi + compj]))
            gop_2_filt_vox[compi + compj] = \
                np.real(np.fft.ifftn(gop_2_filt_dft_vox[compi + compj]))
            gop_0_freq_filt_vox[compi + compj] = \
                np.real(np.fft.ifftn(gop_0_freq_filt_dft_vox[compi + compj]))
    # Return
    return [gop_1_filt_vox, gop_2_filt_vox, gop_0_freq_filt_vox]
#
#                                               Cluster interaction tensor discrete integral
# ==========================================================================================
# Perform the discrete integral over the spatial domain of material cluster I required to
# compute the cluster interaction tensor between the material cluster I and a material
# cluster J. Store the resulting fourth-order tensor in matricial form
def discretecitintegral(comp_order, cluster_filter, gop_1_filt_vox, gop_2_filt_vox,
                        gop_0_freq_filt_vox):
    # Initialize discrete integral
    cit_1_integral_mf = np.zeros((len(comp_order), len(comp_order)))
    cit_2_integral_mf = np.zeros((len(comp_order), len(comp_order)))
    cit_0_freq_integral_mf = np.zeros((len(comp_order), len(comp_order)))
    # Loop over matricial form components
    for i in range(len(comp_order)):
        compi = comp_order[i]
        for j in range(len(comp_order)):
            compj = comp_order[j]
            # Perform discrete integral over the spatial domain of material cluster I
            cit_1_integral_mf[i, j] = \
                mop.kelvinfactor(i, comp_order)*mop.kelvinfactor(j, comp_order)*\
                    np.sum(np.multiply(cluster_filter, gop_1_filt_vox[compi + compj]))
            cit_2_integral_mf[i, j] = \
                mop.kelvinfactor(i, comp_order)*mop.kelvinfactor(j, comp_order)*\
                    np.sum(np.multiply(cluster_filter, gop_2_filt_vox[compi + compj]))
            cit_0_freq_integral_mf[i, j] = \
                mop.kelvinfactor(i, comp_order)*mop.kelvinfactor(j, comp_order)*\
                    np.sum(np.multiply(cluster_filter, gop_0_freq_filt_vox[compi + compj]))
    # Return
    return [cit_1_integral_mf, cit_2_integral_mf, cit_0_freq_integral_mf]
#
#                                            Update and assemble cluster interaction tensors
# ==========================================================================================
# Update cluster interaction tensors and assemble global cluster interaction matrix
def updassemblecit(problem_dict, mat_prop_ref, Se_ref_matrix, material_phases,
                   n_total_clusters, phase_n_clusters, phase_clusters, cit_1_mf, cit_2_mf,
                   cit_0_freq_mf):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Get reference material Young modulus and Poisson ratio
    E_ref = mat_prop_ref['E']
    v_ref = mat_prop_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator's reference material coefficients
    gop_factor_1 = 1.0/(4.0*miu_ref)
    gop_factor_2 = (lam_ref + miu_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    gop_factor_0_freq = numpy.matlib.repmat(Se_ref_matrix, n_total_clusters,
                                            n_total_clusters)
    # Assemble global material independent cluster interaction matrices
    global_cit_1_mf = assemblecit(material_phases, phase_n_clusters, phase_clusters,
                                  comp_order, cit_1_mf)
    global_cit_2_mf = assemblecit(material_phases, phase_n_clusters, phase_clusters,
                                  comp_order, cit_2_mf)
    global_cit_0_freq_mf = assemblecit(material_phases, phase_n_clusters, phase_clusters,
                                       comp_order, cit_0_freq_mf)
    # Assemble global cluster interaction matrix
    global_cit_mf = np.add(np.add(np.multiply(gop_factor_1, global_cit_1_mf),
                                  np.multiply(gop_factor_2, global_cit_2_mf)),
                           np.multiply(gop_factor_0_freq, global_cit_0_freq_mf))
    # Return
    return global_cit_mf
# ------------------------------------------------------------------------------------------
# Assemble the clustering interaction tensors into a single square matrix, sorted by
# ascending order of material phase and by asceding order of cluster labels within each
# material phase
def assemblecit(material_phases, phase_n_clusters, phase_clusters, comp_order, cit_X_mf):
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
    # Initialize global clustering interaction matrix
    global_cit_X_mf = np.zeros((n_total_clusters*len(comp_order),
                                n_total_clusters*len(comp_order)))
    # Initialize row and column cluster indexes
    jclst = 0
    # Loop over material phases
    for mat_phase_B in material_phases:
        # Loop over material phase B clusters
        for cluster_J in phase_clusters[mat_phase_B]:
            # Initialize row cluster index
            iclst = 0
            # Loop over material phases
            for mat_phase_A in material_phases:
                # Set material phase pair
                mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                # Loop over material phase A clusters
                for cluster_I in phase_clusters[mat_phase_A]:
                    # Set cluster pair
                    cluster_pair = str(cluster_I) + '_' + str(cluster_J)
                    # Set assembling ranges
                    i_init = iclst*len(comp_order)
                    i_end = i_init + len(comp_order)
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    # Assemble cluster interaction tensor
                    global_cit_X_mf[i_init:i_end, j_init:j_end] = \
                        cit_X_mf[mat_phase_pair][cluster_pair]
                    # Increment row cluster index
                    iclst = iclst + 1
            # Increment column cluster index
            jclst = jclst + 1
    # Return
    return global_cit_X_mf
