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
# Tensorial operations
import tensorOperations as top
# Homogenized results output
import homogenizedResultsOutput
# VTK output
import VTKOutput
# Material interface
import material.materialInterface
# Linear elastic constitutive model
import material.models.linear_elastic
#
#                                             Solution of the discretized Lippmann-Schwinger
#                                                  system of nonlinear equilibrium equations
# ==========================================================================================
def onlineStage():
    #                                                                           General data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input data file name and post processing directory
    input_file_name = dirs_dict['input_file_name']
    postprocess_dir = dirs_dict['postprocess_dir']
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
    cit_1_mf = clst_dict['cit_1']
    cit_2_mf = clst_dict['cit_2']
    cit_0_freq_mf = clst_dict['cit_0_freq']
    # Get macroscale loading data
    mac_load_type = macload_dict['mac_load_type']
    mac_load = macload_dict['mac_load']
    mac_load_presctype = macload_dict['mac_load_presctype']
    n_load_increments = macload_dict['n_load_increments']
    # Get self-consistent scheme data
    self_consistent_scheme = scs_dict['self_consistent_scheme']
    scs_max_n_iterations = scs_dict['scs_max_n_iterations']
    scs_conv_tol = scs_dict['scs_conv_tol']
    # Get algorithmic parameters
    max_n_iterations = algpar_dict['max_n_iterations']
    conv_tol = algpar_dict['conv_tol']
    #
    #                                                                        Initializations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize homogenized strain and stress tensors
    hom_strain_mf = np.zeros(len(comp_order))
    hom_strain_old_mf = np.zeros(len(comp_order))
    hom_stress_mf = np.zeros(len(comp_order))
    hom_stress_old_mf = np.zeros(len(comp_order))
    # Initialize clusters state variables dictionaries
    clusters_state = dict()
    clusters_state_old = dict()
    # Initialize clusters state variables
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Initialize state variables
            clusters_state[str(cluster)] = \
                              material.materialInterface('init',copy.deepcopy(problem_dict),
                                                          copy.deepcopy(mat_dict),mat_phase)
            clusters_state_old[str(cluster)] = \
                              material.materialInterface('init',copy.deepcopy(problem_dict),
                                                          copy.deepcopy(mat_dict),mat_phase)
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
    #
    #                                                                 Initial state VTK file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK collection file
    VTKOutput.openVTKCollectionFile(input_file_name,postprocess_dir)
    # Write VTK file associated to the initial state
    VTKOutput.writeVTKMacroLoadIncrement(vtk_dict,dirs_dict,problem_dict,mat_dict,rg_dict,
                                                                 clst_dict,0,clusters_state)
    #
    #                                                      Material clusters elastic tangent
    #                                                                     (Zeliang approach)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elastic tangent (matricial form) associated to each material cluster
    clusters_De_mf = clustersElasticTangent(copy.deepcopy(problem_dict),material_properties,
                                                             material_phases,phase_clusters)
    #
    #                                                            Macroscale incremental loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set the incremental macroscale load data
    inc_mac_load_mf,n_presc_mac_stress,presc_strain_idxs,presc_stress_idxs = \
        setMacroscaleLoadIncrements(copy.deepcopy(problem_dict),copy.deepcopy(macload_dict))
    #
    #                                                     Reference material elastic tangent
    #                                                                        (initial guess)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set reference material elastic properties initial guess based on the volume averages
    # of the material phases elastic properties
    material_properties_ref = dict()
    material_properties_ref['E'] = \
                               sum([material_phases_f[phase]*material_properties[phase]['E']
                                                              for phase in material_phases])
    material_properties_ref['v'] = \
                               sum([material_phases_f[phase]*material_properties[phase]['v']
                                                              for phase in material_phases])
    # Compute the reference material elastic tangent (matricial form) and compliance tensor
    # (matrix)
    De_ref_mf,Se_ref_matrix = \
             refMaterialElasticTangents(copy.deepcopy(problem_dict),material_properties_ref)
    #
    #                                                               Incremental loading loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize increment counter
    inc = 1
    # Start incremental loading loop
    while True:
        #                                                        Incremental macroscale load
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain tensor where each component is defined as follows:
        # (a) Incremental macroscale strain (if prescribed macroscale strain component)
        # (b) Incremental homogenized strain (if non-prescribed macroscale strain component)
        inc_mix_strain_mf = np.zeros(len(comp_order))
        inc_mix_strain_mf[presc_strain_idxs] = inc_mac_load_mf['strain'][presc_strain_idxs]
        # Initialize stress tensor where each component is defined as follows:
        # (a) Incremental macroscale stress (if prescribed macroscale stress component)
        # (b) Incremental homogenized stress (if non-prescribed macroscale stress component)
        inc_mix_stress_mf = np.zeros(len(comp_order))
        inc_mix_stress_mf[presc_stress_idxs] = inc_mac_load_mf['stress'][presc_stress_idxs]
        #
        #                                              Self-consistent scheme iterative loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize self-consistent scheme iteration counter
        scs_iter = 0
        # Start self-consistent scheme iterative loop
        while True:
            #
            #                                             Cluster interaction tensors update
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update cluster interaction tensors and assemble global cluster interaction
            # matrix
            global_cit_mf = updateCITs(copy.deepcopy(problem_dict),material_properties_ref,
                                Se_ref_mf,material_phases,n_total_clusters,phase_n_clusters,
                                             phase_clusters,cit_1_mf,cit_2_mf,cit_0_freq_mf)
            #
            #                                                       Global residual matrix 1
            #                                                             (Zeliang approach)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build list which stores the difference between each material cluster elastic
            # tangent (matricial form) and the reference material elastic tangent (matricial
            # form), sorted by ascending order of material phase and by asceding order of
            # cluster labels within each material phase
            diff_De_De_ref_mf = list()
            for mat_phase in material_phases:
                for cluster in phase_clusters[mat_phase]:
                    diff_De_De_ref_mf.append(clusters_De_mf[cluster] - De_ref_mf)
            # Build global matrix similar to the global cluster interaction matrix but where
            # each cluster interaction tensor is double contracted with the difference
            # between the associated material phase elastic tangent and the reference
            # material elastic tangent
            global_cit_De_De_ref_mf = \
                        np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_De_De_ref_mf))
            #
            #                            Cluster incremental strains initial iterative guess
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
            nr_iter = 0
            # Start Newton-Raphson iterative loop
            while True:
                #
                #               Cluster material state update and consistent tangent modulus
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform clusters material state update and compute associated consistent
                # tangent modulus
                clusters_state,clusters_D_mf = \
                                   clustersSUCT(copy.deepcopy(problem_dict),material_phases,
                                        phase_clusters,gbl_inc_strain_mf,clusters_state_old)
                #
                #                                                   Global residual matrix 2
                #                                                         (Zeliang approach)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build list which stores the difference between each material cluster
                # consistent tangent (matricial form) and the material cluster elastic
                # tangent (matricial form), sorted by ascending order of material phase
                # and by ascending order of cluster labels within each material phase
                diff_D_De_mf = list()
                for mat_phase in material_phases:
                    for cluster in phase_clusters[mat_phase]:
                       diff_D_De_mf.append(clusters_D_mf[cluster] - clusters_De_mf[cluster])
                # Build global matrix similar to the global cluster interaction matrix but
                # where each cluster interaction tensor is double contracted with the
                # difference between the associated material phase consistent tangent
                # and elastic tangent
                global_cit_D_De_mf = \
                             np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_D_De_mf))
                #
                #                                                   Global residual matrix 3
                #                                                         ( My approach :) )
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build list which stores the difference between each material cluster
                # consistent tangent (matricial form) and the reference material elastic
                # tangent (matricial form), sorted by ascending order of material phase
                # and by ascending order of cluster labels within each material phase
                diff_D_De_mf = list()
                for mat_phase in material_phases:
                    for cluster in phase_clusters[mat_phase]:
                        diff_D_De_ref_mf.append(D_tensors_mf[mat_phase] - De_tensor_mf_ref)
                # Build global matrix similar to the global cluster interaction matrix but
                # where each cluster interaction tensor is double contracted with the
                # difference between the associated material cluster consistent tangent
                # and the reference material elastic tangent
                global_cit_D_De_ref_mf = \
                             np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_D_De_mf))
                #
                #                                      Incremental homogenized stress tensor
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute the incremental homogenized stress tensor (matricial form) if
                # there are prescribed macroscale stress components
                if n_presc_mac_stress > 0:
                    # Compute homogenized stress tensor (matricial form)
                    _,hom_stress_mf = \
                                homogenizedStrainStressTensors(problem_dict,material_phases,
                                                   phase_clusters,clusters_f,clusters_state)
                    # Compute incremental homogenized stress tensor
                    inc_hom_stress_mf = hom_stress_mf - hom_stress_old_mf
                #
                #                                   Discretized Lippmann-Schwinger system of
                #                                  nonlinear equilibrium equations residuals
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build discretized Lippmann-Schwinger system of nonlinear equilibrium
                # equations residuals
                arg = inc_hom_stress_mf if n_presc_mac_stress > 0 else None
                residual = buildResidual(copy.deepcopy(problem_dict),n_total_clusters,
                                     n_presc_mac_stress,presc_stress_idxs,gbl_inc_strain_mf,
                                                 global_cit_De_De_ref_mf,global_cit_D_De_mf,
                                                    inc_mix_strain_mf,inc_mix_stress_mf,arg)
                #
                #                                                     Convergence evaluation
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute error serving to check iterative convergence
                # error = ...
                # Control Newton-Raphson iteration loop flow
                if error < conv_tol:
                    # Leave Newton-Raphson iterative loop (converged solution)
                    break
                elif nr_iter == max_n_iterations:
                    # Maximum number of Newton-Raphson iterations reached
                    print('error')
                else:
                    # Increment iteration counter
                    nr_iter = nr_iter + 1
                #
                #                                   Discretized Lippmann-Schwinger system of
                #                                   nonlinear equilibrium equations Jacobian
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build discretized Lippmann-Schwinger system of nonlinear equilibrium
                # equations Jacobian
                Jacobian = buildJacobian(copy.deepcopy(problem_dict),n_total_clusters,
                                n_presc_mac_stress,presc_stress_idxs,global_cit_D_De_ref_mf,
                                                                   clusters_f,clusters_D_mf)
                #
                #                                   Discretized Lippmann-Schwinger system of
                #                                           linearized equilibrium equations
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve Lippmann-Schwinger system of linearized equilibrium equations
                d_iter = numpy.linalg.solve(Jacobian,-residual)
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
            #                              Incremental homogenized strain and stress tensors
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute homogenized strain and stress tensors (matricial form)
            hom_strain_mf,hom_stress_mf = \
                                homogenizedStrainStressTensors(problem_dict,material_phases,
                                                   phase_clusters,clusters_f,clusters_state)
            # Compute incremental homogenized strain and stress tensors (matricial form)
            inc_hom_strain_mf = hom_strain_mf - hom_strain_old_mf
            inc_hom_stress_mf = hom_stress_mf - hom_stress_old_mf
            # Assemble the incremental homogenized strain and stress tensors non-prescribed
            # components
            inc_mix_strain_mf[presc_stress_idxs] = inc_hom_strain_mf[presc_stress_idxs]
            inc_mix_stress_mf[presc_strain_idxs] = inc_hom_stress_mf[presc_strain_idxs]
            #
            #                                                         Self-consistent scheme
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update reference material elastic properties through a given self-consistent
            # scheme
            E_ref,v_ref = SCS_UpdateRefMatElasticProperties(self_consistent_scheme,n_dim,
                                             comp_order,inc_mix_strain_mf,inc_mix_stress_mf)
            #
            #                                                         Convergence evaluation
            #                                                       (self-consistent scheme)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute iterative variation of the reference material Young modulus and
            # Poisson ratio
            d_E_ref = E_ref - material_properties_ref['E']
            d_v_ref = v_ref - material_properties_ref['v']
            # Compute metric serving to check self-consistent scheme iterative convergence
            # scs_iter_change = ...
            # Control self-consistent scheme iteration loop flow
            if scs_iter_change < scs_conv_tol:
                # Leave self-consistent scheme iterative loop (converged solution)
                break
            elif scs_iter == scs_max_n_iterations:
                # Maximum number of self-consistent scheme iterations reached
                print('error')
            else:
                # Update reference material elastic properties
                material_properties_ref['E'] = E_ref
                material_properties_ref['v'] = v_ref
                # Increment self-consistent scheme iteration counter
                scs_iter = scs_iter + 1
            #
            #                                             Reference material elastic tangent
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the reference material elastic tangent (matricial form) and compliance
            # tensor (matrix)
            De_ref_mf,Se_ref_matrix = \
             refMaterialElasticTangents(copy.deepcopy(problem_dict),material_properties_ref)
        #
        #                                              Homogenized strain and stress tensors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build homogenized strain tensor
        hom_strain = top.getTensorFromMatricialForm(hom_strain_mf,n_dim,comp_order)
        # Build homogenized stress tensor
        hom_stress = top.getTensorFromMatricialForm(hom_stress_mf,n_dim,comp_order)
        # Compute homogenized out-of-plane stress component in a 2D plane strain problem /
        # strain component in a 2D plane stress problem (output purpose only)
        if problem_type == 1:
            hom_stress_33 = outofplaneHomogenizedComp(problem_type,material_phases,
                                phase_clusters,clusters_f,clusters_state,clusters_state_old)
        #
        #                                  Write increment homogenized results to .hres file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize homogenized results dictionary
        hom_results = dict()
        # Build homogenized results dictionary
        hom_results['hom_strain'] = hom_strain
        hom_results['hom_stress'] = hom_stress
        if problem_type == 1:
            hom_results['hom_stress_33'] = hom_stress_33
        # Write increment homogenized results to associated output file (.hres)
        homogenizedResultsOutput.writeHomResFile(hres_file_path,problem_type,inc,
                                                                                hom_results)
        #
        #                                                                 Increment VTK file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file associated to a given macroscale loading increment
        VTKOutput.writeVTKMacroLoadIncrement(vtk_dict,dirs_dict,problem_dict,mat_dict,
                                                       rg_dict,clst_dict,inc,clusters_state)
        #
        #                                                          Converged state variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the last increment converged state variables
        clusters_state_old = copy.deepcopy(clusters_state)
        # Update the last increment converged homogenized strain and stress tensors
        # (matricial form)
        hom_strain_old_mf = hom_strain_mf
        hom_stress_old_mf = hom_stress_mf
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return if the last macroscale loading increment was completed successfuly,
        # otherwise increment the increment counter
        if inc == n_load_increments:
            # Close VTK collection file
            VTKOutput.closeVTKCollectionFile(input_file_name,postprocess_dir)
            # Finish online stage
            return
        else:
            inc = inc + 1
#
#                                                                    Complementary functions
# ==========================================================================================
# Compute the elastic tangent (matricial form) associated to each material cluster
def clustersElasticTangent(problem_dict,material_properties,material_phases,phase_clusters):
    # Initialize dictionary with the clusters elastic tangent (matricial form)
    clusters_De_mf = dict()
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Compute elastic tangent
            consistent_tangent_mf = \
               linear_elastic.ct(copy.deepcopy(problem_dict),material_properties[mat_phase])
            # Store material cluster elastic tangent
            clusters_De_mf[cluster] = consistent_tangent_mf
    # Return
    return clusters_De_mf
# ------------------------------------------------------------------------------------------
# Set the incremental macroscale load data
def setMacroscaleLoadIncrements(problem_dict,macload_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get macroscale loading data
    mac_load_type = macload_dict['mac_load_type']
    mac_load = macload_dict['mac_load']
    mac_load_presctype = macload_dict['mac_load_presctype']
    n_load_increments = macload_dict['n_load_increments']
    # Set incremental macroscale loading
    inc_mac_load_mf = dict()
    load_types = {1:['strain',],2:['stress',],3:['strain','stress']}
    for load_type in load_types[mac_load_type]:
        inc_mac_load_mf[load_type] = \
                setIncMacLoadMF(n_dim,comp_order,mac_load[load_type][:,1])/n_load_increments
    # Compute number of prescribed macroscale stress components
    n_presc_mac_stress = sum([mac_load_presctype[comp] == 'stress' for comp in comp_order])
    # Set macroscale strain and stress prescribed components indexes
    presc_strain_idxs = list()
    presc_stress_idxs = list()
    for i in range(len(comp_order)):
        comp = comp_order[i]
        if mac_load_presctype[comp] == 'strain':
            presc_strain_idxs.append(i)
        else:
            presc_stress_idxs.append(i)
    # Return
    return [inc_mac_load_mf,n_presc_mac_stress,presc_strain_idxs,presc_stress_idxs]
#
# Under a small strain formulation, set the incremental macroscopic load strain or stress
# tensor matricial form according to Kelvin notation
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
# ------------------------------------------------------------------------------------------
# Compute the reference material elastic tangent (matricial form) and compliance tensor
# (matrix)
def refMaterialElasticTangents(problem_dict,material_properties_ref):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Compute reference material elastic tangent (matricial form)
    De_ref_mf = linear_elastic.ct(copy.deepcopy(problem_dict),material_properties_ref)
    # Compute reference material compliance tensor (matricial form)
    Se_ref_mf = np.linalg.inv(De_ref_mf)
    # Store reference material compliance tensor in a matrix similar to matricial form
    # but without any associated coefficients
    Se_ref_matrix = np.zeros(Se_ref_mf.shape)
    for i in range(len(comp_order)):
        comp = comp_order[i]
        index = tuple([int(j) for j in comp])
        Se_ref_matrix[index] = (1.0/top.kelvinFactor(i,comp_order))*Se_ref_mf
    # Return
    return [De_ref_mf,Se_ref_matrix]
# ------------------------------------------------------------------------------------------
# Perform clusters material state update and compute associated consistent tangent modulus
def clustersSUCT(problem_dict,material_phases,phase_clusters,gbl_inc_strain_mf,
                                                                        clusters_state_old):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Initialize material cluster strain range indexes
    i_init = 0
    i_end = i_init + len(comp_order)
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Get material cluster incremental strain (matricial form)
            inc_strain_mf = gbl_inc_strain_mf[i_init,i_end]
            # Build material cluster incremental strain tensor
            inc_strain = \
                  top.getTensorFromMatricialForm(inc_strain_mf,n_dim,comp_order)
            # Get material cluster last increment converged state variables
            state_variables_old = \
                                 copy.deepcopy(clusters_state_old[str(cluster)])
            # Perform material cluster state update and compute associated
            # consistent tangent modulus
            state_variables,consistent_tangent_mf = \
                        material.materialInterface('suct',problem_dict,mat_dict,
                                       mat_phase,inc_strain,state_variables_old)
            # Store material cluster updated state variables and consistent
            # tangent modulus
            clusters_state[str(cluster)] = state_variables
            clusters_D_mf[str(cluster)] = consistent_tangent_mf
            # Update cluster strain range indexes
            i_init = i_init + len(comp_order)
            i_end = i_init + len(comp_order)
    # Return
    return [clusters_state,clusters_D_mf]
# ------------------------------------------------------------------------------------------
# Update cluster interaction tensors and assemble global cluster interaction matrix
def updateCITs(problem_dict,material_properties_ref,Se_ref_mf,material_phases,
          n_total_clusters,phase_n_clusters,phase_clusters,cit_1_mf,cit_2_mf,cit_0_freq_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get reference material Young modulus and Poisson ratio
    E_ref = material_properties_ref['E']
    v_ref = material_properties_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator's reference material coefficients
    Gop_factor_1 = 1.0/(4.0*miu_ref)
    Gop_factor_2 = (lam_ref + miu_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    Gop_factor_0_freq = numpy.matlib.repmat(Se_ref_mf,n_total_clusters,n_total_clusters)
    # Assemble global material independent cluster interaction matrices
    global_cit_1_mf = assembleCIT(material_phases,phase_n_clusters,phase_clusters,
                                                                        comp_order,cit_1_mf)
    global_cit_2_mf = assembleCIT(material_phases,phase_n_clusters,phase_clusters,
                                                                        comp_order,cit_2_mf)
    global_cit_0_freq_mf = assembleCIT(material_phases,phase_n_clusters,
                                                    phase_clusters,comp_order,cit_0_freq_mf)
    # Assemble global cluster interaction matrix
    global_cit_mf = Gop_factor_1*global_cit_1_mf + Gop_factor_2*global_cit_2_mf + \
                                         np.multiply(Gop_factor_0_freq,global_cit_0_freq_mf)
    # Return
    return global_cit_mf
#
# Assemble the clustering interaction tensors into a single square matrix, sorted by
# ascending order of material phase and by asceding order of cluster labels within each
# material phase
def assembleCIT(material_phases,phase_n_clusters,phase_clusters,comp_order,cit_X_mf):
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
    # Initialize global clustering interaction matrix
    global_cit_X_mf = \
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
                    global_cit_X_mf[i_init:i_end,j_init,j_end] = \
                                                      cit_X_mf[mat_phase_pair][cluster_pair]
                    # Increment row cluster index
                    iclst = iclst + 1
            # Increment column cluster index
            jclst = jclst + 1
    # Return
    return global_cit_X_mf
# ------------------------------------------------------------------------------------------
# Compute residuals of the discretized Lippmann-Schwinger system of nonlinear equilibrium
# equations
def buildResidual(problem_dict,n_total_clusters,n_presc_mac_stress,presc_stress_idxs,
             gbl_inc_strain_mf,global_cit_De_De_ref_mf,global_cit_D_De_mf,inc_mix_strain_mf,
                                                                   inc_mix_stress_mf,*args):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Initialize residual vector
    residual = np.zeros(n_total_clusters*len(comp_order) + n_presc_mac_stress)
    # Compute clusters equilibrium residuals
    residual[0:n_total_clusters*len(comp_order)] = gbl_inc_strain_mf + \
                                    np.matmul(global_cit_De_De_ref_mf,gbl_inc_strain_mf) + \
                                         np.matmul(global_cit_D_De_mf,gbl_inc_strain_mf) - \
                         numpy.matlib.repmat(inc_mix_strain_mf['strain'],1,n_total_clusters)
    # Compute additional residual if there are prescribed macroscale stress components
    if n_presc_mac_stress > 0:
        # Get incremental homogenized stress tensor (matricial form)
        inc_hom_stress_mf = args[0]
        # Compute prescribed macroscale stress components residual
        residual[n_total_clusters*len(comp_order):] = \
                 inc_hom_stress_mf[presc_stress_idxs] - inc_mix_stress_mf[presc_stress_idxs]
    # Return
    return residual
# ------------------------------------------------------------------------------------------
# Compute Jacobian matrix of the discretized Lippmann-Schwinger system of nonlinear
# equilibrium equations
def buildJacobian(problem_dict,n_total_clusters,n_presc_mac_stress,presc_stress_idxs,
                                           global_cit_D_De_ref_mf,clusters_f,clusters_D_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set fourth-order identity tensor (matricial form)
    _,FOId,_,_,_,_,_ = top.setIdentityTensors(n_dim)
    FOId_mf = top.setTensorMatricialForm(FOId,n_dim,comp_order)
    # Initialize Jacobian matrix
    Jacobian = np.zeros(2*(n_total_clusters + n_presc_mac_stress,))
    # Compute Jacobian matrix component solely related with the clusters equilibrium
    # residuals
    i_init = 0
    i_end = n_total_clusters*len(comp_order)
    j_init = 0
    j_end = n_total_clusters*len(comp_order)
    Jacobian[i_init:i_end,j_init:j_end] = \
                      scipy.linalg.block_diag(*(n_total_clusters*[FOId_mf])) + \
                                                          global_cit_D_De_ref_mf
    # Compute Jacobian matrix components arising due to the prescribed macroscale stress
    # components
    if n_presc_mac_stress > 0:
        # Compute Jacobian matrix component related with the clusters equilibrium residuals
        i_init = 0
        i_end = n_total_clusters*len(comp_order)
        j_init = n_total_clusters*len(comp_order)
        j_end = n_total_clusters*len(comp_order) + len(comp_order)
        Jacobian[i_init:i_end,j_init:j_end] = \
                   numpy.matlib.repmat(-1.0*FOId_mf[:,presc_stress_idxs],n_total_clusters,1)
        # Compute Jacobian matrix component related with the prescribed macroscale stress
        # components
        i_init = n_total_clusters*len(comp_order)
        i_end = n_total_clusters*len(comp_order) + len(comp_order)
        j_init = 0
        j_end = n_total_clusters*len(comp_order)
        for mat_phase in material_phases:
            for cluster in phase_clusters[mat_phase]:
                aux_sum = aux_sum + clusters_f[str(cluster)]*clusters_D_mf[cluster]
        Jacobian[i_init:i_end,j_init:j_end] = aux_sum
    # Return
    return Jacobian
# ------------------------------------------------------------------------------------------
# Compute homogenized strain and stress tensors (matricial form)
def homogenizedStrainStressTensors(problem_dict,material_phases,phase_clusters,clusters_f,
                                                                            clusters_state):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Initialize incremental homogenized strain and stress tensors (matricial form)
    inc_hom_strain_mf = np.zeros(len(comp_order))
    inc_hom_stress_mf = np.zeros(len(comp_order))
    # Loop over material phases
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Get material cluster strain and stress tensor (matricial form)
            strain_mf = clusters_state[str(cluster)]['strain_mf']
            stress_mf = clusters_state[str(cluster)]['stress_mf']
            # Add material cluster contribution to homogenized strain and stress tensors
            # (matricial form)
            hom_strain_mf = hom_strain_mf + clusters_f[str(cluster)]*strain_mf
            hom_stress_mf = hom_stress_mf + clusters_f[str(cluster)]*stress_mf
    # Return
    return [hom_strain_mf,hom_stress_mf]
# ------------------------------------------------------------------------------------------
# Compute homogenized out-of-plane strain or stress component in 2D plane strain and plane
# stress problems (output purpose only)
def outofplaneHomogenizedComp(problem_type,material_phases,phase_clusters,clusters_f,
                                                         clusters_state,clusters_state_old):
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
# ------------------------------------------------------------------------------------------
# Update reference material elastic properties through a given self-consistent scheme
def SCS_UpdateRefMatElasticProperties(self_consistent_scheme,problem_dict,
                                                       inc_mix_strain_mf,inc_mix_stress_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set second-order identity tensor
    SOId,_,_,_,_,_,_ = top.setIdentityTensors(n_dim)
    SOId_mf = top.setTensorMatricialForm(SOId,n_dim,comp_order)
    # Perform self-consistent scheme to update the reference material elastic properties
    # 1. Regression-based scheme
    # 2. Projection-based scheme
    if self_consistent_scheme == 1:
        # Initialize self-consistent scheme system of linear equations coefficient matrix
        # and right-hand side
        scs_matrix = np.zeros((2,2))
        scs_rhs = np.zeros(2)
        # Get incremental strain and stress tensors containing the associated macroscale
        # prescribed components and the homogenized components
        inc_mix_strain = top.getTensorFromMatricialForm(inc_mix_strain_mf,n_dim,comp_order)
        inc_mix_stress = top.getTensorFromMatricialForm(inc_mix_stress_mf,n_dim,comp_order)
        # Compute self-consistent scheme system of linear equations right-hand side
        scs_rhs[0] = np.trace(inc_mix_stress)
        scs_rhs[1] = top.ddot22_1(inc_mix_stress,inc_mix_strain)
        # Compute self-consistent scheme system of linear equations coefficient matrix
        scs_matrix[0,0] = np.trace(inc_mix_strain)*np.trace(SOId)
        scs_matrix[0,1] = 2.0*np.trace(inc_mix_strain)
        scs_matrix[1,0] = np.trace(inc_mix_strain)**2
        scs_matrix[1,1] = 2.0*top.ddot22_1(inc_mix_strain,inc_mix_strain)
        # Solve self-consistent scheme system of linear equations
        scs_solution = numpy.linalg.solve(scs_matrix,scs_rhs)
        # Get reference material Lamé parameters
        lam_ref = scs_solution[0]
        miu_ref = scs_solution[1]
        # Compute reference material Young modulus and Poisson ratio
        E_ref = (miu_ref*(3.0*lam_ref + 2.0*miu_ref))/(lam_ref + miu_ref)
        v_ref = lam_ref/(2.0*(lam_ref + miu_ref))
    # Return
    return [E_ref,v_ref]
