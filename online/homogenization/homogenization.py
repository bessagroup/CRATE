#
# Homogenization Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the computation of the homogenized strain and stress fields as well
# as the computation of the effective tangent modulus.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
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
#                                                          Strain/Stress homogenized tensors
# ==========================================================================================
# Compute homogenized strain and stress tensors (matricial form)
def homstatetensors(problem_dict, material_phases, phase_clusters, clusters_f,
                    clusters_state):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
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
    # Return
    return [hom_strain_mf, hom_stress_mf]
# ------------------------------------------------------------------------------------------
# Compute homogenized out-of-plane strain or stress component in 2D plane strain and plane
# stress problems (output purpose only)
def homoutofplanecomp(problem_type, material_phases, phase_clusters, clusters_f,
                      clusters_state, clusters_state_old):
    # Set out-of-plane stress component (2D plane strain problem) / strain component
    # (2D plane stress problem)
    if problem_type == 1:
        comp_name = 'stress_33'
    elif problem_type == 2:
        comp_name = 'strain_33'
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
    # Return
    return hom_comp
#
#                                                                  Effective tangent modulus
# ==========================================================================================
# Compute effective tangent modulus
def efftanmod(problem_dict, material_phases, n_total_clusters, phase_clusters, clusters_f,
              clusters_D_mf, global_cit_D_De_ref_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set second-order identity tensor
    _, _, _, fosym, _, _, _ = top.getidoperators(n_dim)
    FOSym_mf = mop.gettensormf(fosym, n_dim, comp_order)
    # Compute cluster strain concentration tensors system of linear equations coefficient
    # matrix (derivatives of clusters equilibrium residuals)
    csct_matrix = np.add(scipy.linalg.block_diag(*(n_total_clusters*[FOSym_mf,])),
                         global_cit_D_De_ref_mf)
    # Compute cluster strain concentration tensors system of linear equations coefficient
    # right-hand side
    csct_rhs = numpy.matlib.repmat(FOSym_mf, n_total_clusters, 1)
    # Initialize system solution matrix
    csct_solution = np.zeros((n_total_clusters*len(comp_order), len(comp_order)))
    # Solve cluster strain concentration tensors system of linear equations
    for i in range(len(comp_order)):
        csct_solution[:, i] = numpy.linalg.solve(csct_matrix, csct_rhs[:, i])
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
    # Return
    return eff_tangent_mf
