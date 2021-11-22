#
# Cluster Interaction Tensors Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Operations required to compute or manipulate the cluster interaction tensors.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Nov 2020 | Updated documentation.
# Bernardo P. Ferreira | Nov 2020 | Merged update and assemble methods.
# Bernardo P. Ferreira | Nov 2020 | Migrated cluster interaction tensors direct computation
#                                 | methods to CRVE class.
# Bernardo P. Ferreira | Oct 2021 | Extension to finite strains.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
import numpy.matlib
# Shallow and deep copy operations
import copy
# Generate efficient iterators
import itertools as it
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
def gop_material_independent_terms(strain_formulation, n_dim, rve_dims, n_voxels_dims,
                                   comp_order_sym, comp_order_nsym):
    '''Compute Green operator material independent terms in the frequency domain.

    Parameters
    ----------
    strain_formulation: str, {'infinitesimal', 'finite'}
        Problem number of spatial dimensions.
    n_dim : int
        Problem dimension.
    rve_dims : list
        RVE size in each dimension.
    n_voxels_dims : list
        Number of voxels in each dimension of the regular grid (spatial discretization of
        the RVE).
    comp_order_sym : list
        Strain/Stress components symmetric order.
    comp_order_nsym : list
        Strain/Stress components nonsymmetric order.

    Returns
    -------
    gop_1_dft_vox : dict
        Regular grid shaped matrix (item, ndarray) containing each fourth-order matricial
        form component (key, str) of the first Green operator material independent term
        in the frequency domain (discrete Fourier transform).
    gop_2_dft_vox : dict
        Regular grid shaped matrix (item, ndarray) containing each fourth-order matricial
        form component (key, str) of the second Green operator material independent term
        in the frequency domain (discrete Fourier transform).
    gop_0_freq_dft_vox : dict
        Regular grid shaped matrix (item, ndarray) containing each fourth-order matricial
        form component (key, str) of the Green operator zero-frequency (material
        independent) term in the frequency domain (discrete Fourier transform).

    Notes
    -----
    The Green operator is here assumed to be associated with an isotropic elastic reference
    material.

    The material independent terms of the Green operator are conveniently computed to
    perform an efficient update of the Green operator if the associated reference material
    elastic properties are updated by any means (e.g., self-consistent scheme).
    '''
    # Set strain/stress components order according to problem strain formulation
    if strain_formulation == 'infinitesimal':
        comp_order = comp_order_sym
    elif strain_formulation == 'finite':
        comp_order = comp_order_nsym
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = setdiscretefreq(n_dim, rve_dims, n_voxels_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        if ''.join([str(x) for x in fo_idx]) not in var2.keys():
            var2[''.join([str(x) for x in fo_idx])] = \
                np.multiply(np.multiply(var1[fo_idx[0]], var1[fo_idx[1]]),
                            np.multiply(var1[fo_idx[2]], var1[fo_idx[3]]))
        if strain_formulation == 'infinitesimal':
            if str(fo_idx[1]) + str(fo_idx[2]) not in var2.keys():
                var2[str(fo_idx[1]) + str(fo_idx[2])] = \
                    np.multiply(var1[fo_idx[1]], var1[fo_idx[2]])
            if str(fo_idx[0]) + str(fo_idx[2]) not in var2.keys():
                var2[str(fo_idx[0]) + str(fo_idx[2])] = \
                    np.multiply(var1[fo_idx[0]], var1[fo_idx[2]])
            if str(fo_idx[0]) + str(fo_idx[3]) not in var2.keys():
                var2[str(fo_idx[0]) + str(fo_idx[3])] = \
                    np.multiply(var1[fo_idx[0]], var1[fo_idx[3]])
    if n_dim == 2:
        var3 = np.sqrt(np.add(np.square(var1[0]), np.square(var1[1])))
    else:
        var3 = np.sqrt(np.add(np.add(np.square(var1[0]), np.square(var1[1])),
                              np.square(var1[2])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        var4 = [fo_idx[0] == fo_idx[2],]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]),]
        if strain_formulation == 'infinitesimal':
            var4 += [fo_idx[0] == fo_idx[3], fo_idx[1] == fo_idx[3], fo_idx[1] == fo_idx[2]]
            var5 += [str(fo_idx[1]) + str(fo_idx[2]), str(fo_idx[0]) + str(fo_idx[2]),
                     str(fo_idx[0]) + str(fo_idx[3])]
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox]
#
#                                                 Global cluster interaction matrix assembly
# ==========================================================================================
def assemble_cit(strain_formulation, problem_type, mat_prop_ref, Se_ref_matrix,
                 material_phases, phase_n_clusters, phase_clusters, cit_1_mf, cit_2_mf,
                 cit_0_freq_mf):
    '''Update cluster interaction tensors and assemble global cluster interaction matrix.

    Update the cluster interaction tensors by taking into account the material properties
    of the Green operator reference material and assemble them in the global cluster
    interaction matrix.

    Parameters
    ----------
    strain_formulation: str, {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).
    mat_prop_ref : dict
        Reference material properties.
    Se_ref_matrix : ndarray of shape (n_comps, n_comps)
        Reference material compliance tensor. This fourth-order tensor should be stored in a
        matrix but without any coefficients related to the adopted matricial storage method.
    material_phases : list
        RVE material phases labels (str).
    phase_n_clusters : dict
        Number of clusters (item, int) prescribed for each material phase (key, str).
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    cit_1_mf : dict
        Cluster interaction tensors associated with the first Green operator material
        independent term. Each tensor is stored in a dictionary (item, dict) for each pair
        of material phases (key, str), which in turn contains the corresponding matricial
        form (item, ndarray) associated to each pair of clusters (key, str).
    cit_2_mf : dict
        Cluster interaction tensors associated with the second Green operator material
        independent term. Each tensor is stored in a dictionary (item, dict) for each pair
        of material phases (key, str), which in turn contains the corresponding matricial
        form (item, ndarray) associated to each pair of clusters (key, str).
    cit_0_freq_mf : dict
        Cluster interaction tensors associated with the zero-frequency Green operator
        (material independent) term. Each tensor is stored in a dictionary (item, dict) for
        each pair of material phases (key, str), which in turn contains the corresponding
        matricial form (item, ndarray) associated to each pair of clusters (key, str).

    Returns
    -------
    global_cit_mf : ndarray
        Global cluster interaction matrix. Assembly positions are assigned according to the
        order of `material_phases` (1st) and `phase_clusters` (2nd).

    Notes
    -----
    The Green operator is here assumed to be associated with an isotropic elastic reference
    material.
    '''
    # Get problem type parameters
    _, comp_order_sym, comp_order_nsym = \
        mop.get_problem_type_parameters(problem_type)
    # Set strain/stress components order according to problem strain formulation
    if strain_formulation == 'infinitesimal':
        comp_order = comp_order_sym
    elif strain_formulation == 'finite':
        comp_order = comp_order_nsym
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize global clustering interaction matrices
    dims = (n_total_clusters*len(comp_order), n_total_clusters*len(comp_order))
    global_cit_1_mf = np.zeros(dims)
    global_cit_2_mf = np.zeros(dims)
    global_cit_0_freq_mf = np.zeros(dims)
    # Initialize column cluster index
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
                    # Assemble cluster interaction tensors
                    global_cit_1_mf[i_init:i_end, j_init:j_end] = \
                        cit_1_mf[mat_phase_pair][cluster_pair]
                    global_cit_2_mf[i_init:i_end, j_init:j_end] = \
                        cit_2_mf[mat_phase_pair][cluster_pair]
                    global_cit_0_freq_mf[i_init:i_end, j_init:j_end] = \
                        cit_0_freq_mf[mat_phase_pair][cluster_pair]
                    # Increment row cluster index
                    iclst = iclst + 1
            # Increment column cluster index
            jclst = jclst + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble global cluster interaction matrix
    global_cit_mf = np.add(np.add(np.multiply(gop_factor_1, global_cit_1_mf),
                                  np.multiply(gop_factor_2, global_cit_2_mf)),
                           np.multiply(gop_factor_0_freq, global_cit_0_freq_mf))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return global_cit_mf
