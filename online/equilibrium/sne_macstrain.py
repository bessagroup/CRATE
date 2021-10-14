#
# Self-Consistent Scheme Macroscale Strain Formulation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the linearization of the Lippmann-Schwinger nonlinear system of
# equilibrium equations formulated without the farfield strain introduced in the original
# Self-Consistent Clustering Analysis (SCA) reduced order model, i.e. considering the
# macroscale strain tensor naturally arising in the derivation of the Lippmann-Schwinger
# equilibrium equation.
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
#                                                                      Equilibrium residuals
# ==========================================================================================
# Compute residuals of the discretized Lippmann-Schwinger system of nonlinear equilibrium
# equations
def buildresidual(problem_dict, material_phases, phase_clusters, n_total_clusters,
                  n_presc_mac_stress, presc_stress_idxs, global_cit_mf, clusters_state,
                  clusters_state_old, De_ref_mf, gbl_inc_strain_mf, inc_mix_strain_mf,
                  inc_hom_stress_mf, inc_mix_stress_mf):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Initialize clusters incremental polarization stress
    global_inc_pol_stress_mf = np.zeros_like(gbl_inc_strain_mf)
    # Initialize material cluster strain range indexes
    i_init = 0
    i_end = i_init + len(comp_order)
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Compute material cluster incremental stress (matricial form)
            inc_stress_mf = clusters_state[str(cluster)]['stress_mf'] - \
                clusters_state_old[str(cluster)]['stress_mf']
            # Get material cluster incremental strain (matricial form)
            inc_strain_mf = gbl_inc_strain_mf[i_init:i_end]
            # Add cluster incremental polarization stress to global array
            global_inc_pol_stress_mf[i_init:i_end] = inc_stress_mf - \
                np.matmul(De_ref_mf, inc_strain_mf)
            # Update cluster strain range indexes
            i_init = i_init + len(comp_order)
            i_end = i_init + len(comp_order)
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Initialize residual vector
    residual = np.zeros(n_total_clusters*len(comp_order) + n_presc_mac_stress)
    # Compute clusters equilibrium residuals
    residual[0:n_total_clusters*len(comp_order)] = \
        np.subtract(np.add(gbl_inc_strain_mf,
                           np.matmul(global_cit_mf, global_inc_pol_stress_mf)),
                    numpy.matlib.repmat(inc_mix_strain_mf, 1, n_total_clusters))
    # Compute additional residual if there are prescribed macroscale stress components
    if n_presc_mac_stress > 0:
        # Compute prescribed macroscale stress components residual
        residual[n_total_clusters*len(comp_order):] = \
            inc_hom_stress_mf[presc_stress_idxs] - inc_mix_stress_mf[presc_stress_idxs]
    # Return
    return residual
#
#                                                                            Jacobian matrix
# ==========================================================================================
# Compute Jacobian matrix of the discretized Lippmann-Schwinger system of nonlinear
# equilibrium equations
def buildjacobian(problem_dict, material_phases, phase_clusters, n_total_clusters,
                  n_presc_mac_stress, presc_stress_idxs, global_cit_D_De_ref_mf, clusters_f,
                  clusters_D_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set fourth-order identity tensor (matricial form)
    _, _, _, fosym, _, _, _ = top.getidoperators(n_dim)
    fosym_mf = mop.get_tensor_mf(fosym, n_dim, comp_order)
    # Initialize Jacobian matrix
    jacobian = np.zeros(2*(n_total_clusters*len(comp_order) + n_presc_mac_stress,))
    # Compute Jacobian matrix component solely related with the clusters equilibrium
    # residuals
    i_init = 0
    i_end = n_total_clusters*len(comp_order)
    j_init = 0
    j_end = n_total_clusters*len(comp_order)
    jacobian[i_init:i_end, j_init:j_end] = \
        scipy.linalg.block_diag(*(n_total_clusters*[fosym_mf,])) + global_cit_D_De_ref_mf
    # Compute Jacobian matrix components arising due to the prescribed macroscale stress
    # components
    if n_presc_mac_stress > 0:
        # Compute Jacobian matrix component related with the clusters equilibrium residuals
        i_init = 0
        i_end = n_total_clusters*len(comp_order)
        j_init = n_total_clusters*len(comp_order)
        j_end = n_total_clusters*len(comp_order) + len(comp_order)
        jacobian[i_init:i_end, j_init:j_end] = \
            numpy.matlib.repmat(-1.0*fosym_mf[:,presc_stress_idxs], n_total_clusters, 1)
        # Compute Jacobian matrix component related with the prescribed macroscale stress
        # components
        jclst = 0
        for mat_phase in material_phases:
            for cluster in phase_clusters[mat_phase]:
                f_D_mf = clusters_f[str(cluster)]*clusters_D_mf[str(cluster)]
                for k in range(len(presc_stress_idxs)):
                    i = n_total_clusters*len(comp_order) + k
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    jacobian[i, j_init:j_end] = f_D_mf[presc_stress_idxs[k], :]
                # Increment column cluster index
                jclst = jclst + 1
    # Return
    return jacobian
#
#                                                         Equilibrium convergence evaluation
# ==========================================================================================
# Check Newton-Raphson iterative procedure convergence when solving the Lippmann-Schwinger
# nonlinear system of equilibrium equations associated to a given macroscale load
# increment
def checkeqlbconvergence(comp_order, n_total_clusters, inc_mac_load_mf, n_presc_mac_strain,
                       n_presc_mac_stress, presc_strain_idxs, presc_stress_idxs,
                       inc_hom_strain_mf, inc_hom_stress_mf, inc_mix_strain_mf,
                       inc_mix_stress_mf, residual, conv_tol):
    # Initialize criterion convergence flag
    is_converged = False
    # Set strain and stress normalization factors
    if n_presc_mac_strain > 0 and \
            not np.allclose(inc_mac_load_mf['strain'][tuple([presc_strain_idxs])],
            np.zeros(inc_mac_load_mf['strain'][tuple([presc_strain_idxs])].shape),
            atol=1e-10):
        strain_norm_factor = \
            np.linalg.norm(inc_mac_load_mf['strain'][tuple([presc_strain_idxs])])
    elif not np.allclose(inc_hom_strain_mf, np.zeros(inc_hom_strain_mf.shape), atol=1e-10):
        strain_norm_factor = np.linalg.norm(inc_hom_strain_mf)
    else:
        strain_norm_factor = 1
    if n_presc_mac_stress > 0 and not np.allclose(
            inc_mac_load_mf['stress'][tuple([presc_stress_idxs])],
            np.zeros(inc_mac_load_mf['stress'][tuple([presc_stress_idxs])].shape),
            atol=1e-10):
        stress_norm_factor = \
            np.linalg.norm(inc_mac_load_mf['stress'][tuple([presc_stress_idxs])])
    elif not np.allclose(inc_hom_stress_mf, np.zeros(inc_hom_stress_mf.shape), atol=1e-10):
        stress_norm_factor = np.linalg.norm(inc_hom_stress_mf)
    else:
        stress_norm_factor = 1
    # Compute error associated to the clusters equilibrium residuals
    error_A1 = np.linalg.norm(residual[0:n_total_clusters*len(comp_order)])/ \
        strain_norm_factor
    # Compute error associated to the stress homogenization constraints residuals
    if n_presc_mac_stress > 0:
        error_A2 = np.linalg.norm(residual[n_total_clusters*len(comp_order):])/ \
            stress_norm_factor
    # Criterion convergence flag is True if all residual errors converged according to the
    # defined convergence tolerance
    if n_presc_mac_stress == 0:
        error_A2 = None
        is_converged = (error_A1 < conv_tol)
    else:
        is_converged = (error_A1 < conv_tol) and (error_A2 < conv_tol)
    # Compute the normalized error associated to the homogenized strain tensor (relative to
    # the macroscale prescribed strain tensor) for output purposes only
    error_inc_hom_strain = abs((np.linalg.norm(inc_hom_strain_mf) -
                                np.linalg.norm(inc_mix_strain_mf)))/ strain_norm_factor
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [is_converged, error_A1, error_A2, error_inc_hom_strain]
