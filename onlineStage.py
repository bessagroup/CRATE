#
# Online Stage Module (UNNAMED Program)
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
import numpy.matlib
# Shallow and deep copy operations
import copy
# Scientific computation
import scipy.linalg
# Tensorial operations
import tensorOperations as top
# Display errors, warnings and built-in exceptions
import errors
# Linear elastic constitutive model
import linear_elastic


def onlineStage():
    #                                                                           General data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get material data
    material_phases = mat_dict['material_phases']
    material_phases_f = mat_dict['material_phases_f']
    material_properties = mat_dict['material_properties']
    # Get clusters data
    phase_n_clusters = clst_dict['phase_n_clusters']
    phase_clusters = clst_dict['phase_clusters']
    cit_1 = clst_dict['cit_1']
    cit_2 = clst_dict['cit_2']
    cit_0_freq = clst_dict['cit_0_freq']
    # Get macroscale loading data
    mac_load_type = macload_dict['mac_load_type']
    mac_load = macload_dict['mac_load']
    mac_load_presctype = macload_dict['mac_load_presctype']
    n_load_increments = macload_dict['n_load_increments']
    # Get algorithmic parameters
    max_n_iterations = algpar_dict['max_n_iterations']
    conv_tol = algpar_dict['conv_tol']
    #
    #                                                                       Identity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set fourth-order identity tensor
    _,FOId,_,_,_,_,_ = top.setIdentityTensors(n_dim)
    # Set fourth-order identity tensor matricial form
    FOId_mf = top.setTensorMatricialForm(FOId,n_dim,comp_order)
    #
    #                                                     Material phases elasticity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elasticity tensors (matricial form) associated to each material phase
    De_tensors_mf = dict()
    for mat_phase in material_phases:
        # Set required elastic properties according to material phase constitutive model
        req_props = ['E','v']
        for iprop in range(len(req_props)):
            if req_props[iprop] not in material_properties[mat_phase]:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00016',location.filename,location.lineno+1,
                                                                 req_props[iprop],mat_phase)
        # Compute elasticity tensor (matricial form) for current material phase
        De_tensor_mf = np.zeros((len(comp_order),len(comp_order)))
        De_tensor_mf = linear_elastic.linear_elastic_ct(problem_type,n_dim,comp_order,\
                                                             material_properties[mat_phase])
        # Store material phase elasticity tensor (matricial form)
        De_tensors_mf[mat_phase] = De_tensor_mf
    #
    #                                                            Macroscale incremental loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize incremental loading data
    inc_mac_load_mf = dict()
    if mac_load_type == 1:
        inc_mac_load_mf['strain'] = \
                                  setIncMacLoadMF(mac_load['strain'][:,1])/n_load_increments
    elif mac_load_type == 2:
        inc_mac_load_mf['stress'] = \
                                  setIncMacLoadMF(mac_load['stress'][:,1])/n_load_increments
    else:
        inc_mac_load_mf['strain'] = \
                                  setIncMacLoadMF(mac_load['strain'][:,1])/n_load_increments
        inc_mac_load_mf['stress'] = \
                                  setIncMacLoadMF(mac_load['stress'][:,1])/n_load_increments
    # Compute number of prescribed macroscale stress components
    n_presc_mac_stress = sum([inc_mac_load_presc[comp] == 'stress' for comp in comp_order])
    # Set macroscale strain and stress prescribed components indexes
    presc_strain_idxs = list()
    presc_stress_idxs = list()
    for i in range(len(comp_order)):
        comp = comp_order[i]
        if mac_load_presctype[comp] == 'strain':
            presc_strain_idxs.append(i)
        else:
            presc_stress_idxs.append(i)
    # Initialize increment counter
    inc = 1
    # Loop over incremental loads
    while True:
        #                                                        Incremental macroscale load
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain tensor which contains the macroscale prescribed incremental
        # strain components and the homogenized incremental strain components
        # (non-prescribed components). Initialize a similar stress tensor
        inc_mix_strain_mf = np.zeros(len(comp_order))
        inc_mix_strain_mf[presc_strain_idxs] = inc_mac_load_mf['strain'][presc_strain_idxs]
        inc_mix_stress_mf = np.zeros(len(comp_order))
        inc_mix_stress_mf[presc_stress_idxs] = inc_mac_load_mf['stress'][presc_stress_idxs]
        #
        #                                               Reference material elasticity tensor
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material elastic properties initial guess based on the volume
        # averages of the material phases elastic properties
        # GET THE HOMOGENIZED ELASTIC PROPERTIES FROM THE OFFLINE STAGE INSTEAD!
        material_properties_ref = dict()
        material_properties_ref['E'] = \
                               sum([material_phases_f[phase]*material_properties[phase]['E']
                                                              for phase in material_phases])
        material_properties_ref['v'] = \
                               sum([material_phases_f[phase]*material_properties[phase]['v']
                                                              for phase in material_phases])
        # Compute reference material elasticity tensor (matricial form)
        De_tensor_ref_mf = linear_elastic.linear_elastic_ct(problem_type,n_dim,comp_order,\
                                                                    material_properties_ref)
        # Compute reference material compliance tensor (matricial form)
        Se_tensor_mf_ref = np.linalg.inv(De_tensor_ref_mf)
        # Store reference material compliance tensor in a matrix similar to matricial form
        # but without any coefficients associated
        Se_tensor_matrix_ref = np.zeros(Se_tensor_mf_ref.shape)
        for i in range(len(comp_order)):
            comp = comp_order[i]
            index = tuple([int(j) for j in comp])
            Se_tensor_matrix_ref[index] = \
                                       (1.0/top.kelvinFactor(i,comp_order))*Se_tensor_mf_ref
        #
        #                                              Self-consistent scheme iterative loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize self-consistent scheme iteration counter
        scs_iter = 0
        while True:
            #
            #                                             Cluster interaction tensors update
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get reference material Young modulus and Poisson coeficient
            E_ref = material_properties_ref['E']
            v_ref = material_properties_ref['v']
            # Compute reference material Lamé parameters
            lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
            miu_ref = E_ref/(2.0*(1.0 + v_ref))
            # Compute Green operator's reference material coefficients
            Gop_factor_1 = 1.0/(4.0*miu_ref)
            Gop_factor_2 = (lambda_ref + miu_ref)/(miu_ref*(lambda_ref + 2.0*miu_ref))
            Gop_factor_0_freq = numpy.matlib.repmat(Se_tensor_mf_ref,n_total_clusters) # < FIX
            # Assemble global material independent cluster interaction matrices
            global_cit_1_mf = assembleCIT(material_phases,phase_n_clusters,phase_clusters,
                                                                           comp_order,cit_1)
            global_cit_2_mf = assembleCIT(material_phases,phase_n_clusters,phase_clusters,
                                                                           comp_order,cit_2)
            global_cit_0_freq_mf = assembleCIT(material_phases,phase_n_clusters,
                                                       phase_clusters,comp_order,cit_0_freq)
            # Get total number of clusters
            n_total_clusters = \
                         sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
            # Assemble global cluster interaction matrix
            global_cit_mf = Gop_factor_1*global_cit_1_mf + Gop_factor_2*global_cit_2_mf + \
                                                      Gop_factor_0_freq*global_cit_0_freq_mf
            #
            #                                                       Global residual matrix 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build list which stores the difference between each material phase elasticity
            # tensor (matricial form) and the reference material elasticity tensor
            # (matricial form), sorted by ascending order of material phase and by asceding
            # order of cluster labels within each material phase
            diff_De_De_ref_mf = list()
            for mat_phase in material_phases:
                for cluster in phase_clusters[mat_phase]:
                    diff_De_De_ref_mf.append(De_tensors_mf[mat_phase] - De_tensor_mf_ref)
            # Build global matrix similar to the global cluster interaction matrix but where
            # each cluster interaction tensor is double contracted with the difference
            # between the associated material phase elasticity tensor and the reference
            # material elasticity tensor
            global_cit_De_De_ref_mf = \
                        np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_De_De_ref_mf))
            #
            #                                    Incremental strains initial iterative guess
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set clusters strain initial iterative guess
            gbl_inc_strain_mf = np.zeros((n_total_clusters*len(comp_order)))
            # Set incremental homogenized components initial iterative guess
            inc_mix_strain_mf[presc_stress_idxs] = 0.0
            inc_mix_stress_mf[presc_strain_idxs] = 0.0
            #
            #                                                  Newton-Raphson iterative loop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Newton-Raphson iteration counter
            iter = 0
            while True:
                #
                #               Cluster material state update and consistent tangent modulus
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over material phases
                for mat_phase in material_phases:
                    # Loop over material phase clusters
                    for mat_cluster in phase_clusters[mat_phase]:
                        # Get material cluster incremental strain
                        # inc_strain = ...
                        # Get material cluster last increment converged state variables
                        state_variables_old = copy.deepcopy(clusters_state_old
                        # Perform material cluster state update and compute associated
                        # consistent tangent modulus
                        state_variables,consistent_tangent = \
                              materialInterface(problem_dict,mat_dict,mat_phase,mat_cluster,
                                                             inc_strain,state_variables_old)
                        # Store material cluster updated state variables and consistent
                        # tangent modulus
                        clusters_state[str(mat_cluster)] = state_variables
                        clusters_consistent_tangent[str(mat_cluster)] = \
                                                                       consistent_tangent_mf
                #
                #                                                   Global residual matrix 2
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build list which stores the difference between each material phase
                # consistent tangent tensor (matricial form) and the material phase
                # elasticity tensor (matricial form), sorted by ascending order of material
                # phase and by ascending order of cluster labels within each material phase
                diff_D_De_mf = list()
                for mat_phase in material_phases:
                    for cluster in phase_clusters[mat_phase]:
                        diff_D_De_mf.append(D_tensors_mf[mat_phase] -
                                                                   De_tensors_mf[mat_phase])
                # Build global matrix similar to the global cluster interaction matrix but
                # where each cluster interaction tensor is double contracted with the
                # difference between the associated material phase consistent tangent tensor
                # and elasticity tensor
                global_cit_D_De_mf = \
                             np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_D_De_mf))
                #
                #                                                   Global residual matrix 3
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build list which stores the difference between each material phase
                # consistent tangent tensor (matricial form) and the reference material
                # elasticity tensor (matricial form), sorted by ascending order of material
                # phase and by ascending order of cluster labels within each material phase
                diff_D_De_mf = list()
                for mat_phase in material_phases:
                    for cluster in phase_clusters[mat_phase]:
                        diff_D_De_ref_mf.append(D_tensors_mf[mat_phase] - De_tensor_mf_ref)
                # Build global matrix similar to the global cluster interaction matrix but
                # where each cluster interaction tensor is double contracted with the
                # difference between the associated material phase consistent tangent tensor
                # and the reference material elasticity tensor
                global_cit_D_De_ref_mf = \
                             np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_D_De_mf))
                #
                #                                                         Homogenized stress
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if n_presc_mac_stress > 0:
                    # Initialize homogenized stress tensor (matricial form)
                    inc_hom_stress_mf = np.zeros(len(comp_order))
                    # Initialize cluster counter
                    i = 0
                    # Loop over material phases
                    for mat_phase in material_phases:
                        # Loop over material phase clusters
                        for cluster in phase_clusters[mat_phase]:
                            # Get cluster stress tensor (matricial form)
                            i_init = i*len(comp_order)
                            i_end = i*len(comp_order) + len(comp_order)
                            inc_stress_mf = gbl_inc_stress_mf[i_init:i_end]
                            # Add material phase cluster contribution to homogenized stress
                            # tensors (matricial form)
                            inc_hom_stress_mf = inc_hom_stress_mf + \
                                                          clusters_f[str(cluster)]*stress_mf
                            # Increment cluster counter
                            i = i + 1
                    # Assemble homogenized stress tensor in the associated mixed matricial
                    # form
                    inc_mix_stress_mf[presc_strain_idxs] = \
                                                        inc_hom_stress_mf[presc_strain_idxs]
                #
                #                                                  Global residual functions
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize residual vector
                residual = np.zeros(n_total_clusters*len(comp_order) + n_presc_mac_stress)
                # Compute clusters residuals
                residual[0:n_total_clusters*len(comp_order)] = gbl_inc_strain_mf + \
                                    np.matmul(global_cit_De_De_ref_mf,gbl_inc_strain_mf) + \
                                    np.matmul(global_cit_D_De_mf,gbl_inc_strain_mf) - \
                         numpy.matlib.repmat(inc_mix_strain_mf['strain'],1,n_total_clusters)
                # Compute macroscale stress residual
                if n_presc_mac_stress > 0:
                    residual[n_total_clusters*len(comp_order):] = \
                                                    inc_hom_stress_mf[presc_stress_idxs] - \
                                                    inc_mix_stress_mf[presc_stress_idxs]
                #
                #                                                     Convergence evaluation
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Work in progress here...
                #
                #                                                            Jacobian matrix
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize Jacobian matrix
                Jacobian = np.zeros(2*(n_total_clusters + n_presc_mac_stress,))
                # Compute Jacobian matrix component 11
                i_init = 0
                i_end = n_total_clusters*len(comp_order)
                j_init = 0
                j_end = n_total_clusters*len(comp_order)
                Jacobian[i_init:i_end,j_init:j_end] = \
                                  scipy.linalg.block_diag(*(n_total_clusters*[FOId_mf])) + \
                                                                      global_cit_D_De_ref_mf
                # Compute macroscale loading related Jacobian matrix components
                if n_presc_mac_stress > 0:
                    # Compute Jacobian matrix component 12
                    i_init = 0
                    i_end = n_total_clusters*len(comp_order)
                    j_init = n_total_clusters*len(comp_order)
                    j_end = n_total_clusters*len(comp_order) + len(comp_order)
                    Jacobian[i_init:i_end,j_init:j_end] = \
                                      numpy.matlib.repmat(-1.0*FOId_mf[:,presc_stress_idxs],
                                                                         n_total_clusters,1)
                    # Compute Jacobian matrix component 21
                    i_init = n_total_clusters*len(comp_order)
                    i_end = n_total_clusters*len(comp_order) + len(comp_order)
                    j_init = 0
                    j_end = n_total_clusters*len(comp_order)
                    Jacobian[i_init:i_end,j_init:j_end] = \
                              np.sum([material_phases_f[mat_phase]*D_tensors_mf[mat_phase] \
                                     for mat_phase in material_phases])[presc_stress_idxs,:]
                #
                #                                                 System of linear equations
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve Lippmann-Schwinger system of linear equilibrium equations
                d_iter = numpy.linalg.solve(Jacobian,residual)
                #
                #                                       Incremental strains iterative update
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update clusters incremental strain
                gbl_inc_strain_mf = gbl_inc_strain_mf + \
                                                  d_iter[0:n_total_clusters*len(comp_order)]
                # Update homogenized incremental strain components
                if n_presc_mac_stress > 0:
                    inc_mix_strain_mf[presc_stress_idxs] = \
                                                   d_iter[n_total_clusters*len(comp_order):]
            #
            #                                                         Self-consistent scheme
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update reference material elastic properties
            # ...

            # Compute reference material elasticity tensor (matricial form)



    #                                      Write increment homogenized results to .hres file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #                                                                     Increment VTK file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                                                    Complementary functions
# ==========================================================================================
# Assemble the clustering interaction tensors into a single square matrix, sorted by
# ascending order of material phase and by asceding order of cluster labels within each
# material phase
def assembleCIT(material_phases,phase_n_clusters,phase_clusters,comp_order,cit_X):
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
    # Initialize global clustering interaction matrix
    global_cit_X = \
       np.zeros((n_total_clusters*len(comp_order),n_total_clusters*len(comp_order)))
    # Initialize row and column cluster indexes
    jclst = 0
    # Loop over material phases
    for mat_phase_B in material_phases:
        # Loop over material phase B clusters
        for clusterJ in phase_clusters[mat_phase_B]:
            # Initialize row cluster index
            iclst = 0
            # Loop over material phases
            for mat_phase_A in material_phases:
                # Set material phase pair
                mat_phase_pair = mat_phase_A + mat_phase_B
                # Loop over material phase A clusters
                for clusterI in phase_clusters[mat_phase_A]:
                    # Set cluster pair
                    cluster_pair = str(clusterI)+str(clusterJ)
                    # Set assembling ranges
                    i_init = iclst*len(comp_order)
                    i_end = i_init + len(comp_order)
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    # Assemble cluster interaction tensor
                    global_cit_X[i_init:i_end,j_init,j_end] = \
                                                         cit_X[mat_phase_pair][cluster_pair]
                    # Increment row cluster index
                    iclst = iclst + 1
            # Increment column cluster index
            jclst = jclst + 1
    # Return
    return global_cit_X
# ------------------------------------------------------------------------------------------
# Under a small strain formulation, set the incremental macroscopic load matricial form
# according to Kelvin notation
def setIncMacLoadMF(n_dim,comp_order,inc_mac_load_vector):
    # Build incremental macroscale load tensor
    k = 0
    for j in range(n_dim):
        for i in range(n_dim):
            inc_mac_load[i,j] = inc_mac_load_vector[k]
            k = k + 1
    # Set incremental macroscopic load matricial form
    inc_mac_load_mf = top.setTensorMatricialForm(inc_mac_load,n_dim,comp_order)
    # Return
    return inc_mac_load_mf
