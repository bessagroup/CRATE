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
# Python object serialization
import pickle
# Inspect file name and line
import inspect
# Shallow and deep copy operations
import copy
# Display messages
import info
# Generate efficient iterators
import itertools as it
# Display errors, warnings and built-in exceptions
import errors
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
# of the different material phases, i.e. '11', '12', '13', '21', '22', '23', '31', '32',
# '33'. Taking for example the pair '12' and assuming that the material phase '1' contains
# the clusters '0', '1' and '2', and the material phase '2' the clusters '3' and '4', the
# clustering interaction tensors of the pair '12' are stored as
#
#   dictionary['12'] = {'03':cit_mf , '04':cit_mf, '13':cit_mf, '14:cit_mf', ...},
#
#                      where 'ij':cit_mf is the matricial form of the cluster interaction
#                      tensor between the clusters i and j
#
def computeClusterInteractionTensors(dirs_dict,problem_dict,mat_dict,rg_dict,clst_dict):
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
            cit_1_mf[mat_phase_A+mat_phase_B] = dict()
            cit_2_mf[mat_phase_A+mat_phase_B] = dict()
            cit_0_freq_mf[mat_phase_A+mat_phase_B] = dict()
    # Compute Green operator material independent terms
    Gop_1_DFT_vox, Gop_2_DFT_vox, Gop_0_freq_DFT_vox = \
                         GreenOperatorMatIndTerms(n_dim,rve_dims,comp_order,n_voxels_dims)
    # Loop over material phases
    for mat_phase_B in material_phases:
        # ----------------------------------------------------------------------------------
        # Validation:
        if False:
            print('\nMaterial phase B:',mat_phase_B)
        # ----------------------------------------------------------------------------------
        # Loop over material phase B clusters
        for clusterJ in phase_clusters[mat_phase_B]:
            # ------------------------------------------------------------------------------
            # Validation:
            if False:
                print('\n  Material phase B - Cluster J:',clusterJ)
            # ------------------------------------------------------------------------------
            # Set material phase B cluster characteristic function
            _,clusterJ_filter_DFT = getClusterFilter(clusterJ,voxels_clusters)
            # ------------------------------------------------------------------------------
            # Validation:
            if False and mat_phase_B == '1' and clusterJ == 0:
                print('\nmat_phase_B:',mat_phase_B,'clusterJ:',clusterJ)
                print('\nclusterJ_filter_DFT:\n')
                print(clusterJ_filter_DFT)
            # ------------------------------------------------------------------------------
            # Perform discrete convolution between the material phase B cluster
            # characteristic function and each of Green operator material independent
            # terms
            Gop_1_filt_vox,Gop_2_filt_vox,Gop_0_freq_filt_vox = \
                                  discreteCITConvolutionJ(comp_order,rve_dims,n_voxels_dims,
                         clusterJ_filter_DFT,Gop_1_DFT_vox,Gop_2_DFT_vox,Gop_0_freq_DFT_vox)
            # ------------------------------------------------------------------------------
            # Validation:
            if False and mat_phase_B == '2' and clusterJ == 2:
                val_voxel_idx = (2,1,3)
                print('\nmat_phase_B:',mat_phase_B,'clusterJ:',clusterJ)
                print('\nGop_1_filt_vox[' + str(val_voxel_idx) + ']:\n')
                for i in range(len(comp_order)):
                    compi = comp_order[i]
                    for j in range(len(comp_order)):
                        compj = comp_order[j]
                        print('  Component' + compi + compj + ':', \
                             '{:>11.4e}'.format(Gop_1_filt_vox[compi+compj][val_voxel_idx]))
                print('\nGop_2_filt_vox[' + str(val_voxel_idx) + ']:\n')
                for i in range(len(comp_order)):
                    compi = comp_order[i]
                    for j in range(len(comp_order)):
                        compj = comp_order[j]
                        print('  Component' + compi + compj + ':', \
                             '{:>11.4e}'.format(Gop_2_filt_vox[compi+compj][val_voxel_idx]))
                print('\nGop_0_freq_filt_vox[' + str(val_voxel_idx) + ']:\n')
                for i in range(len(comp_order)):
                    compi = comp_order[i]
                    for j in range(len(comp_order)):
                        compj = comp_order[j]
                        print('  Component' + compi + compj + ':', \
                        '{:>11.4e}'.format(Gop_0_freq_filt_vox[compi+compj][val_voxel_idx]))
            # ------------------------------------------------------------------------------
            # Loop over material phases
            for mat_phase_A in material_phases:
                # --------------------------------------------------------------------------
                # Validation:
                if False:
                    print('\n    Material phase A:',mat_phase_A)
                # --------------------------------------------------------------------------
                # Set material phase pair dictionary
                mat_phase_pair = mat_phase_A+mat_phase_B
                # Loop over material phase A clusters
                for clusterI in phase_clusters[mat_phase_A]:
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False:
                        print('\n      Material phase A - Cluster I:',clusterI)
                    # ----------------------------------------------------------------------
                    # Set material phase A cluster characteristic function
                    clusterI_filter,_ = getClusterFilter(clusterI,voxels_clusters)
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False and mat_phase_B == '2' and clusterJ == 2 and \
                                                       mat_phase_A == '2' and clusterI == 2:
                        print('\nmat_phase_B:',mat_phase_B,'clusterJ:',clusterJ)
                        print('mat_phase_A:',mat_phase_A,'clusterI:',clusterI)
                        print('\nclusterI_filter:')
                        print(clusterI_filter.flatten('F'))
                    # ----------------------------------------------------------------------
                    # Perform discrete integral over the spatial domain of material phase A
                    # cluster I
                    cit_1_integral_mf,cit_2_integral_mf,cit_0_freq_integral_mf = \
                                            discreteCITIntegralI(comp_order,clusterI_filter,
                                          Gop_1_filt_vox,Gop_2_filt_vox,Gop_0_freq_filt_vox)
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False and mat_phase_B == '2' and clusterJ == 2 and \
                                                       mat_phase_A == '2' and clusterI == 2:
                        print('\nmat_phase_B:',mat_phase_B,'clusterJ:',clusterJ)
                        print('mat_phase_A:',mat_phase_A,'clusterI:',clusterI)
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
                    cit_1_pair_mf = \
                                 (1.0/(clusters_f[str(clusterI)]*rve_vol))*cit_1_integral_mf
                    cit_2_pair_mf = \
                                 (1.0/(clusters_f[str(clusterI)]*rve_vol))*cit_2_integral_mf
                    cit_0_freq_pair_mf = \
                            (1.0/(clusters_f[str(clusterI)]*rve_vol))*cit_0_freq_integral_mf
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False and mat_phase_B == '2' and clusterJ == 2 and \
                                                       mat_phase_A == '2' and clusterI == 2:
                        # Divide by number of clusters instead of cluster volume fraction
                        # (only on this particular cluster interaction tensor!)
                        cit_1_pair_mf = \
                                 (1.0/np.sum(voxels_clusters == clusterI))*cit_1_integral_mf
                        cit_2_pair_mf = \
                                 (1.0/np.sum(voxels_clusters == clusterI))*cit_2_integral_mf
                        cit_0_freq_pair_mf = \
                            (1.0/np.sum(voxels_clusters == clusterI))*cit_0_freq_integral_mf
                        print('mat_phase_B:',mat_phase_B,'clusterJ:',clusterJ)
                        print('mat_phase_A:',mat_phase_A,'clusterI:',clusterI)
                        print('\ncit_1_pair_mf\n')
                        print(cit_1_pair_mf)
                        print('\ncit_2_pair_mf\n')
                        print(cit_2_pair_mf)
                        print('\ncit_0_freq_pair_mf\n')
                        print(cit_0_freq_pair_mf)
                    # ----------------------------------------------------------------------
                    # Store cluster interaction tensor between material phase A cluster and
                    # material phase B cluster
                    cluster_pair = str(clusterI) + str(clusterJ)
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
        cit_file = open(cit_file_path,'wb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayException(location.filename,location.lineno+1,message)
    # Dump clustering interaction tensors
    info.displayInfo('5','Storing cluster interaction tensors file (.cti)...')
    pickle.dump([cit_1_mf,cit_2_mf,cit_0_freq_mf],cit_file)
    # Close file
    cit_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return
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
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    # Return
    return freqs_dims
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
def GreenOperatorMatIndTerms(n_dim,rve_dims,comp_order,n_voxels_dims):
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = setDiscreteFrequencies(n_dim,rve_dims,n_voxels_dims)
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
                Gop_0_freq_DFT_vox[comp][freq_idx] = 1.0
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
    # --------------------------------------------------------------------------------------
    # Validation:
    if False:
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(precision=4)
        val_voxel_idx = (2,1,3)
        val_voxel_freqs = [freqs_dims[i][val_voxel_idx[i]] for i in range(n_dim)]
        val_voxel_freqs_norm = np.linalg.norm(val_voxel_freqs)
        print('\nGreen operator term 1 components (freq_idx = ' + \
                                                                str(val_voxel_idx) + '):\n')
        print('  Frequency point = ', val_voxel_freqs)
        print('  Norm            = ', '{:>11.4e}'.format(val_voxel_freqs_norm))
        for i in range(len(mf_indexes)):
            comp = ''.join([str(x+1) for x in fo_indexes[i]])
            print('  Component ' + comp + ': ', \
                                '{:>11.4e}'.format(Gop_1_DFT_vox[comp][val_voxel_idx]))
        print('\nGreen operator term 2 components (freq_idx = ' + \
                                                                str(val_voxel_idx) + '):\n')
        print('  Frequency point = ', val_voxel_freqs)
        print('  Norm            = ', '{:>11.4e}'.format(val_voxel_freqs_norm))
        for i in range(len(mf_indexes)):
            comp = ''.join([str(x+1) for x in fo_indexes[i]])
            print('  Component ' + comp + ': ', \
                                '{:>11.4e}'.format(Gop_2_DFT_vox[comp][val_voxel_idx]))
        print('\nGreen operator term 3 components (freq_idx = ' + str((0,0,0)) + '):\n')
        print('  Frequency point = ', val_voxel_freqs)
        print('  Norm            = ', '{:>11.4e}'.format(val_voxel_freqs_norm))
        for i in range(len(mf_indexes)):
            comp = ''.join([str(x+1) for x in fo_indexes[i]])
            print('  Component ' + comp + ': ', \
                                '{:>11.4e}'.format(Gop_0_freq_DFT_vox[comp][(0,0,0)]))
    # --------------------------------------------------------------------------------------
    # Return
    return [Gop_1_DFT_vox,Gop_2_DFT_vox,Gop_0_freq_DFT_vox]
# ------------------------------------------------------------------------------------------
# Set the discrete characteristic function associated to a given material cluster, which in
# the spatial domain is equal to 1 for the material cluster domain points and 0 otherwise.
# Also set the Discrete Fourier Transform (DFT) of the discrete characteristic function
def getClusterFilter(cluster,voxels_clusters):
    # Check if valid cluster
    if not isinstance(cluster,int) and not isinstance(cluster,np.integer):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00040',location.filename,location.lineno+1)
    elif cluster not in voxels_clusters:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00041',location.filename,location.lineno+1)
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
# and each of Green operator material independent terms (frequency domain). Return the
# convolution in the spatial domain by performing an Inverse Discrete Fourier Transform
# (IDFT)
def discreteCITConvolutionJ(comp_order,rve_dims,n_voxels_dims,cluster_filter_DFT,
                                            Gop_1_DFT_vox,Gop_2_DFT_vox,Gop_0_freq_DFT_vox):
    # Initialize discrete convolution (spatial and frequency domain)
    Gop_1_filt_DFT_vox = copy.deepcopy(Gop_1_DFT_vox)
    Gop_2_filt_DFT_vox = copy.deepcopy(Gop_2_DFT_vox)
    Gop_0_freq_filt_DFT_vox = copy.deepcopy(Gop_0_freq_DFT_vox)
    Gop_1_filt_vox = copy.deepcopy(Gop_1_DFT_vox)
    Gop_2_filt_vox = copy.deepcopy(Gop_1_DFT_vox)
    Gop_0_freq_filt_vox = copy.deepcopy(Gop_1_DFT_vox)
    # Compute RVE volume and total number of voxels
    rve_vol = np.prod(rve_dims)
    n_voxels = np.prod(n_voxels_dims)
    # Loop over Green operator components
    for i in range(len(comp_order)):
        compi = comp_order[i]
        for j in range(len(comp_order)):
            compj = comp_order[j]
            # Perform discrete convolution in the frequency domain
            Gop_1_filt_DFT_vox[compi+compj] = \
                                          (rve_vol/n_voxels)*np.multiply(cluster_filter_DFT,
                                                            Gop_1_filt_DFT_vox[compi+compj])
            Gop_2_filt_DFT_vox[compi+compj] = \
                                          (rve_vol/n_voxels)*np.multiply(cluster_filter_DFT,
                                                            Gop_2_filt_DFT_vox[compi+compj])
            Gop_0_freq_filt_DFT_vox[compi+compj] = \
                                          (rve_vol/n_voxels)*np.multiply(cluster_filter_DFT,
                                                       Gop_0_freq_filt_DFT_vox[compi+compj])
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
def discreteCITIntegralI(comp_order,cluster_filter,Gop_1_filt_vox,Gop_2_filt_vox,\
                                                                       Gop_0_freq_filt_vox):
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
                             np.sum(np.multiply(cluster_filter,Gop_1_filt_vox[compi+compj]))
            cit_2_integral_mf[i,j] = \
                             top.kelvinFactor(i,comp_order)*top.kelvinFactor(j,comp_order)*\
                             np.sum(np.multiply(cluster_filter,Gop_2_filt_vox[compi+compj]))
            cit_0_freq_integral_mf[i,j] = \
                             top.kelvinFactor(i,comp_order)*top.kelvinFactor(j,comp_order)*\
                        np.sum(np.multiply(cluster_filter,Gop_0_freq_filt_vox[compi+compj]))
    # Return
    return [cit_1_integral_mf,cit_2_integral_mf,cit_0_freq_integral_mf]
#
#                                                                Perform compatibility check
#                                                (loading previously computed offline stage)
# ==========================================================================================
# Perform a compatibility check between the material phases existent in the spatial
# discretization file and the previously computed loaded cluster interaction tensors
def checkCITCompatibility(mat_dict,clst_dict):
    # Get material phases
    material_phases = mat_dict['material_phases']
    # Check number of cluster interaction tensors
    if len(clst_dict['cit_1_mf'].keys()) != len(material_phases)**2 or \
                          len(clst_dict['cit_2_mf'].keys()) != len(material_phases)**2 or  \
                          len(clst_dict['cit_0_freq_mf'].keys()) != len(material_phases)**2:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00047',location.filename,location.lineno+1,
                                     len(clst_dict['cit_1_mf'].keys()),len(material_phases))
    # Check cluster interaction tensors material phase pairs
    for mat_phase_B in material_phases:
        for mat_phase_A in material_phases:
            mat_phase_pair = mat_phase_A + mat_phase_B
            if mat_phase_pair not in clst_dict['cit_1_mf'].keys() or \
                                    mat_phase_pair not in clst_dict['cit_2_mf'].keys() or  \
                                    mat_phase_pair not in clst_dict['cit_0_freq_mf'].keys():
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00048',location.filename,location.lineno+1,
                                                             mat_phase_pair,material_phases)
