#
# Cluster Interaction Tensors Computation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Computation of the fourth-order cluster interaction tensors (material independent
# components).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Python object serialization
import pickle
# Inspect file name and line
import inspect
# Display messages
import ioput.info as info
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Cluster interaction tensors operations
import cit.citoperations as citop
#
#                                                        Compute cluster interaction tensors
# ==========================================================================================
# Compute the fourth-order cluster interaction tensors between the different material phases
# and store them in matricial form as described below.
#
# Lets assume that the heterogeneous material has 3 phases, namely '1', '2' and '3'. Given
# that the clustering is performed independently on each material phase, the cluster
# interaction tensors must be computed for every pair resulting from the cross combination
# of the different material phases, i.e. '11', '12', '13', '21', '22', '23', '31', '32',
# '33'. Taking for example the pair '12' and assuming that the material phase '1' contains
# the clusters '0', '1' and '2', and the material phase '2' the clusters '3' and '4', the
# clustering interaction tensors of the pair '12' are stored as
#
#   dictionary['12'] = {'03': cit_mf , '04': cit_mf, '13': cit_mf, '14: cit_mf', ...},
#
#                      where 'ij': cit_mf is the matricial form of the cluster interaction
#                      tensor between the clusters i and j
#
def clusterinteractiontensors(dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict):
    # Get directories and paths data
    cit_file_path = dirs_dict['cit_file_path']
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get material data
    material_phases = mat_dict['material_phases']
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    rve_dims = rg_dict['rve_dims']
    # Get clustering data
    phase_clusters = clst_dict['phase_clusters']
    voxels_clusters = clst_dict['voxels_clusters']
    clusters_f = clst_dict['clusters_f']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize cluster interaction tensors dictionaries
    cit_1_mf = dict()
    cit_2_mf = dict()
    cit_0_freq_mf = dict()
    for mat_phase_B in material_phases:
        for mat_phase_A in material_phases:
            cit_1_mf[mat_phase_A + '_' + mat_phase_B] = dict()
            cit_2_mf[mat_phase_A + '_' + mat_phase_B] = dict()
            cit_0_freq_mf[mat_phase_A + '_' + mat_phase_B] = dict()
    # Compute Green operator material independent terms
    gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox = citop.gop_matindterms(
        n_dim, rve_dims, comp_order, n_voxels_dims)
    # Loop over material phases
    for mat_phase_B in material_phases:
        # Loop over material phase B clusters
        for cluster_J in phase_clusters[mat_phase_B]:
            # Set material phase B cluster characteristic function
            _, cluster_J_filter_dft = citop.clusterfilter(cluster_J, voxels_clusters)
            # Perform discrete convolution between the material phase B cluster
            # characteristic function and each of Green operator material independent
            # terms
            gop_1_filt_vox, gop_2_filt_vox, gop_0_freq_filt_vox = \
                citop.clstgopconvolution(comp_order, rve_dims, n_voxels_dims,
                    cluster_J_filter_dft, gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox)
            # Loop over material phases
            for mat_phase_A in material_phases:
                # Set material phase pair dictionary
                mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                # Loop over material phase A clusters
                for cluster_I in phase_clusters[mat_phase_A]:
                    # Set material cluster pair
                    cluster_pair = str(cluster_I) + '_' + str(cluster_J)
                    # Check if cluster-symmetric cluster interaction tensor
                    sym_cluster_pair = switch_pair(cluster_pair)
                    sym_mat_phase_pair = switch_pair(mat_phase_pair)
                    is_clst_sym = sym_cluster_pair in cit_1_mf[sym_mat_phase_pair].keys()
                    # Compute cluster interaction tensor between material phase A cluster
                    # and material phase B cluster (complete computation or
                    # cluster-symmetric computation)
                    if is_clst_sym:
                        # Set cluster volume fractions ratio
                        clst_f_ratio = clusters_f[str(cluster_J)]/clusters_f[str(cluster_I)]
                        # Compute clustering interaction tensor between material phase A
                        # cluster and material phase B cluster through cluster-symmetry
                        cit_1_mf[mat_phase_pair][cluster_pair] = \
                            np.multiply(clst_f_ratio,
                                        cit_1_mf[sym_mat_phase_pair][sym_cluster_pair])
                        cit_2_mf[mat_phase_pair][cluster_pair] = \
                            np.multiply(clst_f_ratio,
                                        cit_2_mf[sym_mat_phase_pair][sym_cluster_pair])
                        cit_0_freq_mf[mat_phase_pair][cluster_pair] = \
                            np.multiply(clst_f_ratio,
                                        cit_0_freq_mf[sym_mat_phase_pair][sym_cluster_pair])
                    else:
                        # Set material phase A cluster characteristic function
                        cluster_I_filter, _ = citop.clusterfilter(cluster_I,
                                                                  voxels_clusters)
                        # Perform discrete integral over the spatial domain of material
                        # phase A cluster I
                        cit_1_integral_mf, cit_2_integral_mf, cit_0_freq_integral_mf = \
                            citop.discretecitintegral(comp_order, cluster_I_filter,
                                                      gop_1_filt_vox, gop_2_filt_vox,
                                                      gop_0_freq_filt_vox)
                        # Compute cluster interaction tensor between the material phase A
                        # cluster and the material phase B cluster
                        rve_vol = np.prod(rve_dims)
                        cit_1_pair_mf = \
                            np.multiply((1.0/(clusters_f[str(cluster_I)]*rve_vol)),
                                        cit_1_integral_mf)
                        cit_2_pair_mf = \
                            np.multiply((1.0/(clusters_f[str(cluster_I)]*rve_vol)),
                                        cit_2_integral_mf)
                        cit_0_freq_pair_mf = \
                            np.multiply((1.0/(clusters_f[str(cluster_I)]*rve_vol)),
                                        cit_0_freq_integral_mf)
                        # Store cluster interaction tensor between material phase A cluster
                        # and material phase B cluster
                        cit_1_mf[mat_phase_pair][cluster_pair] = cit_1_pair_mf
                        cit_2_mf[mat_phase_pair][cluster_pair] = cit_2_pair_mf
                        cit_0_freq_mf[mat_phase_pair][cluster_pair] = cit_0_freq_pair_mf
    # Store clustering interaction tensors
    clst_dict['cit_1_mf'] = cit_1_mf
    clst_dict['cit_2_mf'] = cit_2_mf
    clst_dict['cit_0_freq_mf'] = cit_0_freq_mf
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open file which contains all the required information associated to the clustering
    # discretization
    try:
        cit_file = open(cit_file_path, 'wb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1, message)
    # Dump clustering interaction tensors
    info.displayinfo('5', 'Storing cluster interaction tensors file (.cti)...')
    pickle.dump([cit_1_mf, cit_2_mf, cit_0_freq_mf], cit_file)
    # Close file
    cit_file.close()
# ------------------------------------------------------------------------------------------
# Given a string in the format 'X[X]_[Y]Y', switch the left and right sides and return
# the string 'Y[Y]_[X]X'
def switch_pair(x):
    if not isinstance(x, str) or x.count('_') != 1:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00061', location.filename, location.lineno + 1)
    return '_'.join(x.split('_')[::-1])
