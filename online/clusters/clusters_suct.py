#
# Clusters State Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the state update and consistent tangent modulus of the material
# clusters.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Shallow and deep copy operations
import copy
# Matricial operations
import tensor.matrixoperations as mop
# Material interface
import material.materialinterface
# Linear elastic constitutive model
import material.models.linear_elastic
#
#                                                           Clusters elastic tangent modulus
# ==========================================================================================
# Compute the elastic tangent (matricial form) associated to each material cluster
def clusterselastictanmod(problem_dict, material_properties, material_phases,
                          phase_clusters):
    # Initialize dictionary with the clusters elastic tangent (matricial form)
    clusters_De_mf = dict()
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Compute elastic tangent
            consistent_tangent_mf = material.models.linear_elastic.ct(
                problem_dict, material_properties[mat_phase])
            # Store material cluster elastic tangent
            clusters_De_mf[str(cluster)] = consistent_tangent_mf
    # Return
    return clusters_De_mf
#
#                                       Clusters state update and consistent tangent modulus
# ==========================================================================================
# Perform clusters material state update and compute associated consistent tangent modulus
def clusterssuct(problem_dict, mat_dict, clst_dict, algpar_dict, phase_clusters,
                 gbl_inc_strain_mf, clusters_state_old):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get material data
    material_phases = mat_dict['material_phases']
    # Initialize clusters state variables and consistent tangent
    clusters_state = dict()
    clusters_D_mf = dict()
    # Initialize material cluster strain range indexes
    i_init = 0
    i_end = i_init + len(comp_order)
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Get material cluster incremental strain (matricial form)
            inc_strain_mf = gbl_inc_strain_mf[i_init:i_end]
            # Build material cluster incremental strain tensor
            inc_strain = mop.gettensorfrommf(inc_strain_mf, n_dim, comp_order)
            # Get material cluster last increment converged state variables
            state_variables_old = copy.deepcopy(clusters_state_old[str(cluster)])
            # Perform material cluster state update and compute associated
            # consistent tangent modulus
            state_variables,consistent_tangent_mf = \
                material.materialinterface.materialinterface(
                    'suct', problem_dict, mat_dict, clst_dict, algpar_dict, mat_phase,
                    inc_strain, state_variables_old)
            # Store material cluster updated state variables and consistent
            # tangent modulus
            clusters_state[str(cluster)] = state_variables
            clusters_D_mf[str(cluster)] = consistent_tangent_mf
            # Update cluster strain range indexes
            i_init = i_init + len(comp_order)
            i_end = i_init + len(comp_order)
    # Return
    return [clusters_state, clusters_D_mf]
