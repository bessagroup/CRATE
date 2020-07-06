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
        # ----------------------------------------------------------------------------------
        # Validation:
        if False:
            print('\nMaterial phase B:',mat_phase_B)
        # ----------------------------------------------------------------------------------
        # Loop over material phase B clusters
        for cluster_J in phase_clusters[mat_phase_B]:
            # ------------------------------------------------------------------------------
            # Validation:
            if False:
                print('\n  Material phase B - Cluster J:',cluster_J)
            # ------------------------------------------------------------------------------
            # Set material phase B cluster characteristic function
            _, cluster_J_filter_dft = citop.clusterfilter(cluster_J, voxels_clusters)
            # ------------------------------------------------------------------------------
            # Validation:
            if False and mat_phase_B == '1' and cluster_J == 0:
                print('\nmat_phase_B:',mat_phase_B,'cluster_J:',cluster_J)
                print('\nclusterJ_filter_DFT:\n')
                print(cluster_J_filter_dft)
            # ------------------------------------------------------------------------------
            # Perform discrete convolution between the material phase B cluster
            # characteristic function and each of Green operator material independent
            # terms
            gop_1_filt_vox, gop_2_filt_vox, gop_0_freq_filt_vox = \
                citop.clstgopconvolution(comp_order, rve_dims, n_voxels_dims,
                    cluster_J_filter_dft, gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox)
            # ------------------------------------------------------------------------------
            # Validation:
            if False and mat_phase_B == '2' and cluster_J == 2:
                val_voxel_idx = (2,1,3)
                print('\nmat_phase_B:',mat_phase_B,'cluster_J:',cluster_J)
                print('\nGop_1_filt_vox[' + str(val_voxel_idx) + ']:\n')
                for i in range(len(comp_order)):
                    compi = comp_order[i]
                    for j in range(len(comp_order)):
                        compj = comp_order[j]
                        print('  Component' + compi + compj + ':', \
                             '{:>11.4e}'.format(gop_1_filt_vox[compi+compj][val_voxel_idx]))
                print('\nGop_2_filt_vox[' + str(val_voxel_idx) + ']:\n')
                for i in range(len(comp_order)):
                    compi = comp_order[i]
                    for j in range(len(comp_order)):
                        compj = comp_order[j]
                        print('  Component' + compi + compj + ':', \
                             '{:>11.4e}'.format(gop_2_filt_vox[compi+compj][val_voxel_idx]))
                print('\nGop_0_freq_filt_vox[' + str(val_voxel_idx) + ']:\n')
                for i in range(len(comp_order)):
                    compi = comp_order[i]
                    for j in range(len(comp_order)):
                        compj = comp_order[j]
                        print('  Component' + compi + compj + ':', \
                        '{:>11.4e}'.format(gop_0_freq_filt_vox[compi+compj][val_voxel_idx]))
            # ------------------------------------------------------------------------------
            # Loop over material phases
            for mat_phase_A in material_phases:
                # --------------------------------------------------------------------------
                # Validation:
                if False:
                    print('\n    Material phase A:',mat_phase_A)
                # --------------------------------------------------------------------------
                # Set material phase pair dictionary
                mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                # Loop over material phase A clusters
                for cluster_I in phase_clusters[mat_phase_A]:
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False:
                        print('\n      Material phase A - Cluster I:',cluster_I)
                    # ----------------------------------------------------------------------
                    # Set material phase A cluster characteristic function
                    cluster_I_filter, _ = citop.clusterfilter(cluster_I, voxels_clusters)
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False and mat_phase_B == '2' and cluster_J == 2 and \
                            mat_phase_A == '2' and cluster_I == 2:
                        print('\nmat_phase_B:',mat_phase_B,'cluster_J:',cluster_J)
                        print('mat_phase_A:',mat_phase_A,'cluster_I:',cluster_I)
                        print('\nclusterI_filter:')
                        print(cluster_I_filter.flatten('F'))
                    # ----------------------------------------------------------------------
                    # Perform discrete integral over the spatial domain of material phase A
                    # cluster I
                    cit_1_integral_mf, cit_2_integral_mf, cit_0_freq_integral_mf = \
                        citop.discretecitintegral(comp_order, cluster_I_filter,
                                                   gop_1_filt_vox, gop_2_filt_vox,
                                                   gop_0_freq_filt_vox)
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False and mat_phase_B == '2' and cluster_J == 2 and \
                                                      mat_phase_A == '2' and cluster_I == 2:
                        print('\nmat_phase_B:',mat_phase_B,'cluster_J:',cluster_J)
                        print('mat_phase_A:',mat_phase_A,'cluster_I:',cluster_I)
                        print('\ncit_1_integral_mf\n')
                        print(cit_1_integral_mf)
                        print('\ncit_2_integral_mf\n')
                        print(cit_2_integral_mf)
                        print('\ncit_0_freq_integral_mf\n')
                        print(cit_0_freq_integral_mf)
                    # ----------------------------------------------------------------------
                    # Compute cluster interaction tensor between the material phase A
                    # cluster and the material phase B cluster
                    rve_vol = np.prod(rve_dims)
                    cit_1_pair_mf = np.multiply((1.0/(clusters_f[str(cluster_I)]*rve_vol)),
                                                cit_1_integral_mf)
                    cit_2_pair_mf = np.multiply((1.0/(clusters_f[str(cluster_I)]*rve_vol)),
                                                cit_2_integral_mf)
                    cit_0_freq_pair_mf = \
                        np.multiply((1.0/(clusters_f[str(cluster_I)]*rve_vol)),
                                    cit_0_freq_integral_mf)
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False and mat_phase_B == '2' and cluster_J == 2 and \
                                                      mat_phase_A == '2' and cluster_I == 2:
                        # Divide by number of clusters instead of cluster volume fraction
                        # (only on this particular cluster interaction tensor!)
                        cit_1_pair_mf = \
                                (1.0/np.sum(voxels_clusters == cluster_I))*cit_1_integral_mf
                        cit_2_pair_mf = \
                                (1.0/np.sum(voxels_clusters == cluster_I))*cit_2_integral_mf
                        cit_0_freq_pair_mf = \
                           (1.0/np.sum(voxels_clusters == cluster_I))*cit_0_freq_integral_mf
                        print('mat_phase_B:',mat_phase_B,'cluster_J:',cluster_J)
                        print('mat_phase_A:',mat_phase_A,'cluster_I:',cluster_I)
                        print('\ncit_1_pair_mf\n')
                        print(cit_1_pair_mf)
                        print('\ncit_2_pair_mf\n')
                        print(cit_2_pair_mf)
                        print('\ncit_0_freq_pair_mf\n')
                        print(cit_0_freq_pair_mf)
                    # ----------------------------------------------------------------------
                    # Store cluster interaction tensor between material phase A cluster and
                    # material phase B cluster
                    cluster_pair = str(cluster_I) + '_' + str(cluster_J)
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
