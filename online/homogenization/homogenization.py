#
# Homogenization Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the computation of the homogenized strain and stress tensors and to
# the computation of the CRVE effective tangent modulus.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Nov 2020 | Updated documentation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
import numpy.matlib
# Scientific computation
import scipy.linalg
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                          Homogenized strain/stress tensors
# ==========================================================================================
def homstatetensors(comp_order, material_phases, phase_clusters, clusters_f,
                    clusters_state):
    '''Compute the homogenized strain and stress tensors.

    Parameters
    ----------
    comp_order : list
        Strain/Stress components (str) order.
    material_phases : list
        RVE material phases labels (str).
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    clusters_state : dict
        Material constitutive model state variables (item, dict) associated to each material
        cluster (key, str). Despite the several constitutive model dependent state
        variables, the strain and stress tensors matricial form must be available through
        the keys 'strain_mf' and 'stress_mf', respectively.

    Returns
    -------
    hom_strain_mf : ndarray of shape (n_comps,)
        Homogenized strain tensor in matricial form.
    hom_stress_mf : ndarray of shape (n_comps,)
        Homogenized stress tensor in matricial form.

    Notes
    -----
    The matricial form storage is perform according to the provided strain/stress
    components order.
    '''
    # Initialize incremental homogenized strain and stress tensors (matricial form)
    hom_strain_mf = np.zeros(len(comp_order))
    hom_stress_mf = np.zeros(len(comp_order))
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Get material cluster strain and stress tensor (matricial form)
            strain_mf = clusters_state[str(cluster)]['strain_mf']
            stress_mf = clusters_state[str(cluster)]['stress_mf']
            # Add material cluster contribution to homogenized strain and stress tensors
            # (matricial form)
            hom_strain_mf = np.add(hom_strain_mf, clusters_f[str(cluster)]*strain_mf)
            hom_stress_mf = np.add(hom_stress_mf, clusters_f[str(cluster)]*stress_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [hom_strain_mf, hom_stress_mf]
# ------------------------------------------------------------------------------------------
def homoutofplanecomp(problem_type, material_phases, phase_clusters, clusters_f,
                      clusters_state):
    '''Compute homogenized out-of-plane strain or stress component.

    Parameters
    ----------
    problem_type : {1, 2}
        Plane problem type identifier: (1) Plane strain, (2) Plane stress. This parameter
        determines the nature of the out-of-plain homogenized component that is computed.
    material_phases : list
        RVE material phases labels (str).
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    clusters_state : dict
        Material constitutive model state variables (item, dict) associated to each material
        cluster (key, str). Despite the several constitutive model dependent state
        variables, the non-null out-of-plain strain or stress component must be available
        through the keys 'strain_33' or 'stress_33', respectively.

    Return
    ------
    hom_comp : float
        Homogenized out-of-plane strain or stress component.
    '''
    # Set out-of-plane stress component (2D plane strain problem) / strain component
    # (2D plane stress problem)
    if problem_type == 1:
        comp_name = 'stress_33'
    elif problem_type == 2:
        comp_name = 'strain_33'
    else:
        raise RuntimeError('Unknown plane problem type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize homogenized out-of-plane component
    hom_comp = 0.0
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Add material cluster contribution to the homogenized out-of-plane component
            # component
            hom_comp = hom_comp + \
                clusters_f[str(cluster)]*clusters_state[str(cluster)][comp_name]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return hom_comp
#
#                                                             CRVE effective tangent modulus
# ==========================================================================================
# Compute effective tangent modulus
def efftanmod(n_dim, comp_order, material_phases, phase_clusters, clusters_f, clusters_D_mf,
              global_cit_D_De_ref_mf):
    '''Compute CRVE effective (homogenized) tangent modulus.

    Parameters
    ----------
    n_dim : int
        Problem dimension.
    comp_order : list
        Strain/Stress components (str) order.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    clusters_D_mf : dict
        Consistent tangent modulus (item, ndarray of shape(n_comps, n_comps)) associated to
        each material cluster (key, str).
    global_cit_D_De_ref_mf : ndarray
        Global matrix with the same shape of the global cluster interaction matrix but where
        each cluster interaction tensor (I)(J) is double contracted with the difference
        between the material cluster (J) consistent tangent modulus and the reference
        material elastic tangent modulus (assembled in matricial form).

    Returns
    -------
    eff_tangent_mf : ndarray of shape(n_comps, n_comps)
        CRVE effective (homogenized) tangent modulus in matricial form.

    Notes
    -----
    The matricial form storage is perform according to the provided strain/stress
    components order.
    '''
    # Get total number of clusters
    n_total_clusters = len(clusters_f.keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set fourth-order symmetric projection tensor (matricial form)
    _, _, _, FOsym, _, _, _ = top.getidoperators(n_dim)
    FOSym_mf = mop.gettensormf(FOsym, n_dim, comp_order)
    # Compute equilibrium Jacobian matrix (luster strain concentration tensors system of
    # linear equations coefficient matrix)
    csct_matrix = np.add(scipy.linalg.block_diag(*(n_total_clusters*[FOSym_mf,])),
                         global_cit_D_De_ref_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Select clusters strain concentration tensors computation option:
    #
    # Option 1 - Solve linear system of equations
    #
    # Option 2 - Direct computation from inverse of equilibrium Jacobian matrix
    #
    option = 2
    # OPTION 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if option == 1:
        # Compute cluster strain concentration tensors system of linear equations
        # right-hand side
        csct_rhs = numpy.matlib.repmat(FOSym_mf, n_total_clusters, 1)
        # Initialize system solution matrix (containing clusters strain concentration
        # tensors)
        csct_solution = np.zeros((n_total_clusters*len(comp_order), len(comp_order)))
        # Solve cluster strain concentration tensors system of linear equations
        for i in range(len(comp_order)):
            csct_solution[:, i] = numpy.linalg.solve(csct_matrix, csct_rhs[:, i])
    # OPTION 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    elif option == 2:
        # Compute inverse of equilibrium Jacobian matrix
        csct_matrix_inv = numpy.linalg.inv(csct_matrix)
        # Initialize system solution matrix (containing clusters strain concentration
        # tensors)
        csct_solution = np.zeros((n_total_clusters*len(comp_order), len(comp_order)))
        # Initialize cluster indexes
        i_init = 0
        i_end = i_init + len(comp_order)
        j_init = 0
        j_end = j_init + len(comp_order)
        # Loop over material phases
        for mat_phase_I in material_phases:
            # Loop over material phase clusters
            for cluster_I in phase_clusters[mat_phase_I]:
                # Loop over material phases
                for mat_phase_J in material_phases:
                    # Loop over material phase clusters
                    for cluster_J in phase_clusters[mat_phase_J]:
                        # Add cluster J contribution to cluster I strain concentration
                        # tensor
                        csct_solution[i_init:i_end, :] += \
                            csct_matrix_inv[i_init:i_end, j_init:j_end]
                        # Increment cluster index
                        j_init = j_init + len(comp_order)
                        j_end = j_init + len(comp_order)
                # Increment cluster indexes
                i_init = i_init + len(comp_order)
                i_end = i_init + len(comp_order)
                j_init = 0
                j_end = j_init + len(comp_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize effective tangent modulus
    eff_tangent_mf = np.zeros((len(comp_order), len(comp_order)))
    # Initialize cluster index
    i_init = 0
    i_end = i_init + len(comp_order)
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Get material cluster volume fraction
            cluster_f = clusters_f[str(cluster)]
            # Get material cluster consistent tangent (matricial form)
            cluster_D_mf = clusters_D_mf[str(cluster)]
            # Get material cluster strain concentration tensor
            cluster_sct_mf = csct_solution[i_init:i_end, :]
            # Add material cluster contribution to effective tangent modulus
            eff_tangent_mf = eff_tangent_mf + \
                             cluster_f*np.matmul(cluster_D_mf, cluster_sct_mf)
            # Increment cluster index
            i_init = i_init + len(comp_order)
            i_end = i_init + len(comp_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return eff_tangent_mf
