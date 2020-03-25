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
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
import numpy.matlib
# Inspect file name and line
import inspect
# Date and time
import time
# Shallow and deep copy operations
import copy
# Display messages
import info
# Scientific computation
import scipy.linalg
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
#                                                                          Validation output
# ==========================================================================================
# Set validation flags (terminal output)
#  0. Validation header
#  1. Initializations
#  2. Material clusters elastic tangent
#  3. Macroscale load increments
#  4. Reference material elastic tangent
#  5. Incremental macroscale load
#  6. Cluster incremental strains initial guess
#  7. Cluster interaction tensors update
#  8. Global residual matrix 1 (Zeliang approach)
#  9. Clusters material state update and consistent tangent
# 10. Global residual matrix 2 (Zeliang approach)
# 11. Global residual matrix 3 (My approach)
# 12. Incremental homogenized strain and stress tensors
# 13. Lippmann-Schwinger SNLE residuals
# 14. Convergence evaluation
# 15. Lippmann-Schwinger SNLE Jacobian
# 16. Lippmann-Schwinger SLE solution
# 17. Incremental strains iterative update
# 18. Incremental homogenized strain and stress tensors
# 19. Self-consistent scheme
# 20. Self-consistent scheme convergence evaluation
# 21. Homogenized strain and stress tensors
output_idx = []
is_Validation = [False for i in range(22)]
for i in output_idx:
    is_Validation[i] = True
#
#                                             Solution of the discretized Lippmann-Schwinger
#                                                  system of nonlinear equilibrium equations
# ==========================================================================================
def onlineStage(dirs_dict,problem_dict,mat_dict,rg_dict,clst_dict,macload_dict,scs_dict,
                                                                      algpar_dict,vtk_dict):
    #                                                                           General data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input data file name and post processing directory
    input_file_name = dirs_dict['input_file_name']
    postprocess_dir = dirs_dict['postprocess_dir']
    hres_file_path = dirs_dict['hres_file_path']
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get material data
    material_phases = mat_dict['material_phases']
    material_phases_f = mat_dict['material_phases_f']
    material_properties = mat_dict['material_properties']
    # Get clusters data
    phase_n_clusters = clst_dict['phase_n_clusters']
    phase_clusters = clst_dict['phase_clusters']
    clusters_f = clst_dict['clusters_f']
    cit_1_mf = clst_dict['cit_1_mf']
    cit_2_mf = clst_dict['cit_2_mf']
    cit_0_freq_mf = clst_dict['cit_0_freq_mf']
    # Get macroscale loading data
    mac_load_presctype = macload_dict['mac_load_presctype']
    n_load_increments = macload_dict['n_load_increments']
    # Get self-consistent scheme data
    self_consistent_scheme = scs_dict['self_consistent_scheme']
    scs_max_n_iterations = scs_dict['scs_max_n_iterations']
    scs_conv_tol = scs_dict['scs_conv_tol']
    # Get algorithmic parameters
    max_n_iterations = algpar_dict['max_n_iterations']
    conv_tol = algpar_dict['conv_tol']
    # Get VTK output parameters
    is_VTK_output = vtk_dict['is_VTK_output']
    if is_VTK_output:
        vtk_inc_div = vtk_dict['vtk_inc_div']
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[0]:
        print('\n' + 'Online Stage validation' + '\n' + 23*'-')
        np.set_printoptions(linewidth = np.inf)
        np.set_printoptions(formatter={'float':'{: 11.4e}'.format})
    # --------------------------------------------------------------------------------------
    #
    #                                                                        Initializations
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set online stage initial time
    os_init_time = time.time()
    # Set far-field implementation flag
    is_farfield_implementation = True
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
                                        material.materialInterface.materialInterface('init',
                              copy.deepcopy(problem_dict),copy.deepcopy(mat_dict),mat_phase)
            clusters_state_old[str(cluster)] = \
                                        material.materialInterface.materialInterface('init',
                              copy.deepcopy(problem_dict),copy.deepcopy(mat_dict),mat_phase)
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase] for mat_phase in material_phases])
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[1]:
        section = 'Initializations'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'n_total_clusters = ' + str(n_total_clusters))
    # --------------------------------------------------------------------------------------
    #
    #                                                                 Initial state VTK file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_VTK_output:
        # Open VTK collection file
        VTKOutput.openVTKCollectionFile(input_file_name,postprocess_dir)
        # Write VTK file associated to the initial state
        VTKOutput.writeVTKMacroLoadIncrement(vtk_dict,dirs_dict,problem_dict,mat_dict,
                                                         rg_dict,clst_dict,0,clusters_state)
    #
    #                                                      Material clusters elastic tangent
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elastic tangent (matricial form) associated to each material cluster
    clusters_De_mf = clustersElasticTangent(copy.deepcopy(problem_dict),material_properties,
                                                             material_phases,phase_clusters)
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[2]:
        section = 'Material clusters elastic tangent'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        for cluster in range(n_total_clusters):
            print('\n' + 'clusters_De_mf[cluster = ' + str(cluster) + ']:' + '\n')
            print(clusters_De_mf[str(cluster)])
    # --------------------------------------------------------------------------------------
    #
    #                                                            Macroscale incremental loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set the incremental macroscale load data
    inc_mac_load_mf,n_presc_mac_strain,n_presc_mac_stress,presc_strain_idxs, \
                presc_stress_idxs = setMacroscaleLoadIncrements(copy.deepcopy(problem_dict),
                                                                copy.deepcopy(macload_dict))
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[3]:
        section = 'Macroscale load increments'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        if n_presc_mac_strain > 0:
            print('\n' + 'n_presc_mac_strain = ' + str(n_presc_mac_strain))
            print('\n' + 'presc_strain_idxs = ' + str(presc_strain_idxs))
            print('\n' + 'inc_mac_load_mf[strain]:' + '\n')
            print(inc_mac_load_mf['strain'])
        if n_presc_mac_stress > 0:
            print('\n' + 'n_presc_mac_stress = ' + str(n_presc_mac_stress))
            print('\n' + 'presc_stress_idxs = ' + str(presc_stress_idxs))
            print('\n' + 'inc_mac_load_mf[stress]:' + '\n')
            print(inc_mac_load_mf['stress'])
    # --------------------------------------------------------------------------------------
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
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[4]:
        section = 'Reference material elastic tangent'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'material_properties_ref[\'E\'] = '+ str(material_properties_ref['E']))
        print('\n' + 'material_properties_ref[\'v\'] = '+ str(material_properties_ref['v']))
        print('\n' + 'De_ref_mf:' + '\n')
        print(De_ref_mf)
        print('\n' + 'Se_ref_matrix:' + '\n')
        print(Se_ref_matrix)
    # --------------------------------------------------------------------------------------
    #
    #                                                               Incremental loading loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize increment counter
    inc = 1
    info.displayInfo('7','init',inc)
    # Set increment initial time
    inc_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Validation:
    if any(is_Validation):
        print('\n' + (92 - len(' Increment ' + str(inc)))*'-' + ' Increment ' + str(inc))
    # --------------------------------------------------------------------------------------
    # Start incremental loading loop
    while True:
        #                                                        Incremental macroscale load
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not is_farfield_implementation:
            # Initialize strain tensor where each component is defined as follows:
            # (a) Incremental macroscale strain (if prescribed macroscale strain component)
            # (b) Incremental homogenized strain (if non-prescribed macroscale strain
            # component)
            inc_mix_strain_mf = np.zeros(len(comp_order))
            if n_presc_mac_strain > 0:
                inc_mix_strain_mf[presc_strain_idxs] = \
                                                inc_mac_load_mf['strain'][presc_strain_idxs]
            # Initialize stress tensor where each component is defined as follows:
            # (a) Incremental macroscale stress (if prescribed macroscale stress component)
            # (b) Incremental homogenized stress (if non-prescribed macroscale stress
            #     component)
            inc_mix_stress_mf = np.zeros(len(comp_order))
            if n_presc_mac_stress > 0:
                inc_mix_stress_mf[presc_stress_idxs] = \
                                                inc_mac_load_mf['stress'][presc_stress_idxs]
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[5]:
                section = 'Incremental macroscale load'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                print('\n' + 'inc_mix_strain_mf:' + '\n')
                print(inc_mix_strain_mf)
                print('\n' + 'inc_mix_stress_mf:' + '\n')
                print(inc_mix_stress_mf)
            # ------------------------------------------------------------------------------
        #                                Cluster incremental strains initial iterative guess
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set clusters strain initial iterative guess
        gbl_inc_strain_mf = np.zeros((n_total_clusters*len(comp_order)))
        # Set additional initial iterative guesses
        if is_farfield_implementation:
            # Set far-field strain initial iterative guess
            inc_farfield_strain_mf = np.zeros(len(comp_order))
        else:
            # Set incremental homogenized components initial iterative guess
            inc_mix_strain_mf[presc_stress_idxs] = 0.0
            inc_mix_stress_mf[presc_strain_idxs] = 0.0
        # ----------------------------------------------------------------------------------
        # Validation:
        if is_Validation[6]:
            section = 'Cluster incremental strains initial guess'
            print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
            print('\n' + 'gbl_inc_strain_mf:' + '\n')
            print(gbl_inc_strain_mf)
            if is_farfield_implementation:
                print('\n' + 'inc_farfield_strain_mf:' + '\n')
                print(inc_farfield_strain_mf)
            else:
                print('\n' + 'inc_mix_strain_mf:' + '\n')
                print(inc_mix_strain_mf)
                print('\n' + 'inc_mix_stress_mf:' + '\n')
                print(inc_mix_stress_mf)
        # ----------------------------------------------------------------------------------
        #
        #                                              Self-consistent scheme iterative loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize self-consistent scheme iteration counter
        scs_iter = 0
        info.displayInfo('8','init',scs_iter)
        # Set self-consistent scheme iteration initial time
        scs_iter_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Validation:
        if any(is_Validation):
            print('\n' + (92 - len(' SCS Iteration: ' + str(scs_iter)))*'-' + \
                                                         ' SCS Iteration: ' + str(scs_iter))
        # ----------------------------------------------------------------------------------
        # Start self-consistent scheme iterative loop
        while True:
            #
            #                                             Cluster interaction tensors update
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[7]:
                section = 'Cluster interaction tensors update'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
            # ------------------------------------------------------------------------------
            # Update cluster interaction tensors and assemble global cluster interaction
            # matrix
            global_cit_mf = updateCITs(copy.deepcopy(problem_dict),material_properties_ref,
                            Se_ref_matrix,material_phases,n_total_clusters,phase_n_clusters,
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
                    diff_De_De_ref_mf.append(clusters_De_mf[str(cluster)] - De_ref_mf)
            # Build global matrix similar to the global cluster interaction matrix but where
            # each cluster interaction tensor is double contracted with the difference
            # between the associated material phase elastic tangent and the reference
            # material elastic tangent
            global_cit_De_De_ref_mf = \
                        np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_De_De_ref_mf))
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[8]:
                section = 'Global residual matrix 1 (Zeliang approach)'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                for i in range(len(diff_De_De_ref_mf)):
                    print('\n' + 'diff_De_De_ref_mf[' + str(i) + ']:' + '\n')
                    print(diff_De_De_ref_mf[i])
                print('\n' + 'global_cit_De_De_ref_mf:' + '\n')
                print(global_cit_De_De_ref_mf)
            # ------------------------------------------------------------------------------
            #

            #
            #                                                  Newton-Raphson iterative loop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Newton-Raphson iteration counter
            nr_iter = 0
            info.displayInfo('9','init')
            # Set Newton-Raphson iteration initial time
            nr_iter_init_time = time.time()
            # ------------------------------------------------------------------------------
            # Validation:
            if any(is_Validation):
                print('\n' + (92 - len(' NR Iteration: ' + str(nr_iter)))*'-' + \
                                                          ' NR Iteration: ' + str(nr_iter))
            # ------------------------------------------------------------------------------
            # Start Newton-Raphson iterative loop
            while True:
                #
                #               Cluster material state update and consistent tangent modulus
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform clusters material state update and compute associated consistent
                # tangent modulus
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[9]:
                    section = 'Cluster su and ct'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                # --------------------------------------------------------------------------
                clusters_state,clusters_D_mf = \
                           clustersSUCT(copy.deepcopy(problem_dict),copy.deepcopy(mat_dict),
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
                        diff_D_De_mf.append(clusters_D_mf[str(cluster)] -
                                                               clusters_De_mf[str(cluster)])
                # Build global matrix similar to the global cluster interaction matrix but
                # where each cluster interaction tensor is double contracted with the
                # difference between the associated material phase consistent tangent
                # and elastic tangent
                global_cit_D_De_mf = \
                             np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_D_De_mf))
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[10]:
                    section = 'Global residual matrix 2 (Zeliang approach)'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    for i in range(len(diff_D_De_mf)):
                        print('\n' + 'diff_D_De_mf[' + str(i) + ']:' + '\n')
                        print(diff_D_De_mf[i])
                    print('\n' + 'global_cit_D_De_mf:' + '\n')
                    print(global_cit_D_De_mf)
                # --------------------------------------------------------------------------
                #
                #                                                   Global residual matrix 3
                #                                                         ( My approach :) )
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build list which stores the difference between each material cluster
                # consistent tangent (matricial form) and the reference material elastic
                # tangent (matricial form), sorted by ascending order of material phase
                # and by ascending order of cluster labels within each material phase
                diff_D_De_ref_mf = list()
                for mat_phase in material_phases:
                    for cluster in phase_clusters[mat_phase]:
                        diff_D_De_ref_mf.append(clusters_D_mf[str(cluster)] - De_ref_mf)
                # Build global matrix similar to the global cluster interaction matrix but
                # where each cluster interaction tensor is double contracted with the
                # difference between the associated material cluster consistent tangent
                # and the reference material elastic tangent
                global_cit_D_De_ref_mf = \
                         np.matmul(global_cit_mf,scipy.linalg.block_diag(*diff_D_De_ref_mf))
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[11]:
                    section = 'Global residual matrix 3 (my approach)'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    for i in range(len(diff_D_De_ref_mf)):
                        print('\n' + 'diff_D_De_ref_mf[' + str(i) + ']:' + '\n')
                        print(diff_D_De_ref_mf[i])
                    print('\n' + 'global_cit_D_De_ref_mf:' + '\n')
                    print(global_cit_D_De_ref_mf)
                # --------------------------------------------------------------------------
                #
                #                          Incremental homogenized strain and stress tensors
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute homogenized strain and stress tensors (matricial form)
                hom_strain_mf,hom_stress_mf = \
                                homogenizedStrainStressTensors(problem_dict,material_phases,
                                                   phase_clusters,clusters_f,clusters_state)
                # Compute incremental homogenized strain and stress tensors (matricial form)
                inc_hom_strain_mf = hom_strain_mf - hom_strain_old_mf
                inc_hom_stress_mf = hom_stress_mf - hom_stress_old_mf
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[12]:
                    section = 'Incremental homogenized strain and stress tensors'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    print('\n' + 'hom_strain_mf:' + '\n')
                    print(hom_strain_mf)
                    print('\n' + 'hom_stress_mf:' + '\n')
                    print(hom_stress_mf)
                    print('\n' + 'inc_hom_strain_mf:' + '\n')
                    print(inc_hom_strain_mf)
                    print('\n' + 'inc_hom_stress_mf:' + '\n')
                    print(inc_hom_stress_mf)
                    if is_farfield_implementation:
                        print('\n' + 'inc_farfield_strain_mf:' + '\n')
                        print(inc_farfield_strain_mf)
                    else:
                        print('\n' + 'inc_mix_strain_mf:' + '\n')
                        print(inc_mix_strain_mf)
                        print('\n' + 'inc_mix_stress_mf:' + '\n')
                        print(inc_mix_stress_mf)
                # --------------------------------------------------------------------------
                #                                   Discretized Lippmann-Schwinger system of
                #                                  nonlinear equilibrium equations residuals
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build discretized Lippmann-Schwinger system of nonlinear equilibrium
                # equations residuals
                if is_farfield_implementation:
                    residual = buildResidual2(copy.deepcopy(problem_dict),n_total_clusters,
                                 presc_strain_idxs,global_cit_D_De_ref_mf,inc_hom_strain_mf,
                                        inc_hom_stress_mf,inc_mac_load_mf,gbl_inc_strain_mf,
                                                                     inc_farfield_strain_mf)
                else:
                    arg = inc_hom_stress_mf if n_presc_mac_stress > 0 else None
                    residual = buildResidual(copy.deepcopy(problem_dict),n_total_clusters,
                                     n_presc_mac_stress,presc_stress_idxs,gbl_inc_strain_mf,
                                 global_cit_D_De_ref_mf,inc_mix_strain_mf,inc_mix_stress_mf,
                                                                                        arg)
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[13]:
                    section = 'Lippmann-Schwinger SNLE residuals'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    print('\n' + 'residual:' + '\n')
                    print(residual)
                # --------------------------------------------------------------------------
                #
                #                                                     Convergence evaluation
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check Newton-Raphson iterative procedure convergence
                if is_farfield_implementation:
                    is_converged,error_A1,error_A2,error_A3 = \
                        checkNRConvergence2(comp_order,n_total_clusters,inc_mac_load_mf,
                                    n_presc_mac_strain,n_presc_mac_stress,presc_strain_idxs,
                                      presc_stress_idxs,inc_hom_strain_mf,residual,conv_tol)
                    info.displayInfo('9','iter',nr_iter,time.time() - nr_iter_init_time,
                                                                 error_A1,error_A2,error_A3)
                else:
                    is_converged_A,is_converged_B,error_B1,error_B2 = \
                          checkNRConvergence(comp_order,mac_load_presctype,n_total_clusters,
                                       inc_mac_load_mf,n_presc_mac_stress,inc_hom_strain_mf,
                                                        inc_hom_stress_mf,inc_mix_strain_mf,
                                                        inc_mix_stress_mf,residual,conv_tol)
                    is_converged = is_converged_A and is_converged_B
                # Control Newton-Raphson iteration loop flow
                if is_converged:
                    # Leave Newton-Raphson iterative loop (converged solution)
                    break
                elif nr_iter == max_n_iterations:
                    # Maximum number of Newton-Raphson iterations reached
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayError('E00061',location.filename,location.lineno+1,
                                                     max_n_iterations,inc,error_B1,error_B2)
                else:
                    # Increment iteration counter
                    nr_iter = nr_iter + 1
                    # Set Newton-Raphson iteration initial time
                    nr_iter_init_time = time.time()
                    # ----------------------------------------------------------------------
                    # Validation:
                    if any(is_Validation):
                        print('\n' + (92 - len(' NR Iteration: ' + str(nr_iter)))*'-' + \
                                                           ' NR Iteration: ' + str(nr_iter))
                    # ----------------------------------------------------------------------
                #
                #                                   Discretized Lippmann-Schwinger system of
                #                                   nonlinear equilibrium equations Jacobian
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build discretized Lippmann-Schwinger system of nonlinear equilibrium
                # equations Jacobian
                if is_farfield_implementation:
                    Jacobian = buildJacobian2(problem_dict,material_phases,phase_clusters,
                                  n_total_clusters,presc_strain_idxs,global_cit_D_De_ref_mf,
                                                                   clusters_f,clusters_D_mf)
                else:
                    Jacobian = buildJacobian(copy.deepcopy(problem_dict),material_phases,
                       phase_clusters,n_total_clusters,n_presc_mac_stress,presc_stress_idxs,
                                            global_cit_D_De_ref_mf,clusters_f,clusters_D_mf)
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[15]:
                    section = 'Lippmann-Schwinger SNLE Jacobian'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    print('\n' + 'Jacobian:' + '\n')
                    print(Jacobian[0:6,0:6])
                # --------------------------------------------------------------------------
                #
                #                                   Discretized Lippmann-Schwinger system of
                #                                           linearized equilibrium equations
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve Lippmann-Schwinger system of linearized equilibrium equations
                d_iter = numpy.linalg.solve(Jacobian,-residual)
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[16]:
                    section = 'Lippmann-Schwinger SLE solution'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    print('\n' + 'd_iter:' + '\n')
                    print(d_iter)
                # --------------------------------------------------------------------------
                #
                #                                       Incremental strains iterative update
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update clusters incremental strain
                gbl_inc_strain_mf = gbl_inc_strain_mf + \
                                                  d_iter[0:n_total_clusters*len(comp_order)]
                # Update additional quantities
                if is_farfield_implementation:
                    # Update far-field strain
                    inc_farfield_strain_mf = inc_farfield_strain_mf + \
                                                   d_iter[n_total_clusters*len(comp_order):]
                else:
                    # Update homogenized incremental strain components
                    if n_presc_mac_stress > 0:
                        inc_mix_strain_mf[presc_stress_idxs] = \
                                                   inc_mix_strain_mf[presc_stress_idxs] +  \
                                                   d_iter[n_total_clusters*len(comp_order):]
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[17]:
                    section = 'Incremental strains iterative update'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    print('\n' + 'gbl_inc_strain_mf:' + '\n')
                    print(gbl_inc_strain_mf)
                    if is_farfield_implementation:
                        print('\n' + 'inc_farfield_strain_mf:' + '\n')
                        print(inc_farfield_strain_mf)
                    else:
                        if n_presc_mac_stress > 0:
                            print('\n' + 'inc_mix_strain_mf:' + '\n')
                            print(inc_mix_strain_mf)
                # --------------------------------------------------------------------------
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
            if not is_farfield_implementation:
                if n_presc_mac_stress > 0:
                    inc_mix_strain_mf[presc_stress_idxs] = \
                                                        inc_hom_strain_mf[presc_stress_idxs]
                if n_presc_mac_strain > 0:
                    inc_mix_stress_mf[presc_strain_idxs] = \
                                                        inc_hom_stress_mf[presc_strain_idxs]
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[18]:
                section = 'Incremental homogenized strain and stress tensors'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                print('\n' + 'hom_strain_mf:' + '\n')
                print(hom_strain_mf)
                print('\n' + 'hom_stress_mf:' + '\n')
                print(hom_stress_mf)
                print('\n' + 'inc_hom_strain_mf:' + '\n')
                print(inc_hom_strain_mf)
                print('\n' + 'inc_hom_stress_mf:' + '\n')
                print(inc_hom_stress_mf)
                if is_farfield_implementation:
                    print('\n' + 'inc_farfield_strain_mf:' + '\n')
                    print(inc_farfield_strain_mf)
                else:
                    print('\n' + 'inc_mix_strain_mf:' + '\n')
                    print(inc_mix_strain_mf)
                    print('\n' + 'inc_mix_stress_mf:' + '\n')
                    print(inc_mix_stress_mf)
            # ------------------------------------------------------------------------------
            #
            #                                                         Self-consistent scheme
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update reference material elastic properties through a given self-consistent
            # scheme
            if is_farfield_implementation:
                E_ref,v_ref = SCS_UpdateRefMatElasticProperties(self_consistent_scheme,
                                           problem_dict,inc_hom_strain_mf,inc_hom_stress_mf)
            else:
                E_ref,v_ref = SCS_UpdateRefMatElasticProperties(self_consistent_scheme,
                                           problem_dict,inc_mix_strain_mf,inc_mix_stress_mf)
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[19]:
                section = 'Self-consistent scheme'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                print('\n' + 'E_ref = ' + str(E_ref))
                print('\n' + 'v_ref = ' + str(v_ref))
            # ------------------------------------------------------------------------------
            #
            #                                                         Convergence evaluation
            #                                                       (self-consistent scheme)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check self-consistent scheme iterative procedure convergence
            is_scs_conv_E,is_scs_conv_v,norm_d_E_ref,norm_d_v_ref = \
                       checkSCSConvergence(E_ref,v_ref,material_properties_ref,scs_conv_tol)
            info.displayInfo('8','end',time.time() - scs_iter_init_time)
            # Control self-consistent scheme iteration loop flow
            if is_scs_conv_E and is_scs_conv_v:
                # Leave self-consistent scheme iterative loop (converged solution)
                break
            elif scs_iter == scs_max_n_iterations:
                # Maximum number of self-consistent scheme iterations reached
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00062',location.filename,location.lineno+1,
                                         scs_max_n_iterations,inc,norm_d_E_ref,norm_d_v_ref)
            else:
                # Update reference material elastic properties
                material_properties_ref['E'] = E_ref
                material_properties_ref['v'] = v_ref
                # Increment self-consistent scheme iteration counter
                scs_iter = scs_iter + 1
                info.displayInfo('8','init',scs_iter,norm_d_E_ref,norm_d_v_ref)
                # Set self-consistent scheme iteration initial time
                scs_iter_init_time = time.time()
                # --------------------------------------------------------------------------
                # Validation:
                if any(is_Validation):
                    print('\n' + (92 - len(' SCS Iteration: ' + str(scs_iter)))*'-' + \
                                                         ' SCS Iteration: ' + str(scs_iter))
                # --------------------------------------------------------------------------
            #
            #                                             Reference material elastic tangent
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute the reference material elastic tangent (matricial form) and compliance
            # tensor (matrix)
            De_ref_mf,Se_ref_matrix = \
             refMaterialElasticTangents(copy.deepcopy(problem_dict),material_properties_ref)
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[4]:
                section = 'Reference material elastic tangent'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                print('\n' + 'material_properties_ref[\'E\'] = ' + \
                                                          str(material_properties_ref['E']))
                print('\n' + 'material_properties_ref[\'v\'] = ' + \
                                                          str(material_properties_ref['v']))
                print('\n' + 'De_ref_mf:' + '\n')
                print(De_ref_mf)
                print('\n' + 'Se_ref_matrix:' + '\n')
                print(Se_ref_matrix)
            # ------------------------------------------------------------------------------
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
        # ----------------------------------------------------------------------------------
        # Validation:
        if is_Validation[21]:
            section = 'Homogenized strain and stress tensors'
            print('\n' + 'hom_strain:' + '\n')
            print(hom_strain)
            print('\n' + 'hom_stress:' + '\n')
            print(hom_stress)
            if problem_type == 1:
                print('\n' + 'hom_stress_33' + '{:11.4e}'.format(hom_stress_33))
            if inc == 2 and nr_iter > 0:
                sys.exit(1)
        # ----------------------------------------------------------------------------------
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
        # Write VTK file associated to the macroscale loading increment
        if is_VTK_output and inc % vtk_inc_div == 0:
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
        info.displayInfo('7','end',hom_strain,hom_stress,\
                                     time.time() - inc_init_time,time.time() - os_init_time)
        if inc == n_load_increments:
            # Close VTK collection file
            if is_VTK_output:
                VTKOutput.closeVTKCollectionFile(input_file_name,postprocess_dir)
            # Finish online stage
            return
        else:
            inc = inc + 1
            info.displayInfo('7','init',inc)
            # Set increment initial time
            inc_init_time = time.time()
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
                              material.models.linear_elastic.ct(copy.deepcopy(problem_dict),
                                                             material_properties[mat_phase])
            # Store material cluster elastic tangent
            clusters_De_mf[str(cluster)] = consistent_tangent_mf
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
    # Compute number of prescribed macroscale strain and stress components
    n_presc_mac_strain = sum([mac_load_presctype[comp] == 'strain' for comp in comp_order])
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
    return [inc_mac_load_mf,n_presc_mac_strain,n_presc_mac_stress,presc_strain_idxs,
                                                                          presc_stress_idxs]
#
# Under a small strain formulation, set the incremental macroscopic load strain or stress
# tensor matricial form according to Kelvin notation
def setIncMacLoadMF(n_dim,comp_order,inc_mac_load_vector):
    # Initialize incremental macroscale load tensor
    inc_mac_load = np.zeros((n_dim,n_dim))
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
    De_ref_mf = \
      material.models.linear_elastic.ct(copy.deepcopy(problem_dict),material_properties_ref)
    # Compute reference material compliance tensor (matricial form)
    Se_ref_mf = np.linalg.inv(De_ref_mf)
    # Store reference material compliance tensor in a matrix similar to matricial form
    # but without any associated coefficients
    Se_ref_matrix = np.zeros(Se_ref_mf.shape)
    for j in range(len(comp_order)):
        for i in range(len(comp_order)):
            Se_ref_matrix[i,j] = (1.0/top.kelvinFactor(i,comp_order))*\
                                         (1.0/top.kelvinFactor(j,comp_order))*Se_ref_mf[i,j]
    # Return
    return [De_ref_mf,Se_ref_matrix]
# ------------------------------------------------------------------------------------------
# Perform clusters material state update and compute associated consistent tangent modulus
def clustersSUCT(problem_dict,mat_dict,phase_clusters,gbl_inc_strain_mf,clusters_state_old):
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
            inc_strain = \
                  top.getTensorFromMatricialForm(inc_strain_mf,n_dim,comp_order)
            # Get material cluster last increment converged state variables
            state_variables_old = \
                                 copy.deepcopy(clusters_state_old[str(cluster)])
            # Perform material cluster state update and compute associated
            # consistent tangent modulus
            state_variables,consistent_tangent_mf = \
                  material.materialInterface.materialInterface('suct',problem_dict,mat_dict,
                                                   mat_phase,inc_strain,state_variables_old)
            # Store material cluster updated state variables and consistent
            # tangent modulus
            clusters_state[str(cluster)] = state_variables
            clusters_D_mf[str(cluster)] = consistent_tangent_mf
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[9]:
                print('\n' + 'cluster: ' + str(cluster))
                print('\n' + 'state_variables[\'e_strain_mf\']:' + '\n')
                print(state_variables['e_strain_mf'])
                print('\n' + 'state_variables[\'strain_mf\']:' + '\n')
                print(state_variables['strain_mf'])
                print('\n' + 'state_variables[\'stress_mf\']:' + '\n')
                print(state_variables['stress_mf'])
                if n_dim == 2:
                    print('\n' + 'state_variables[\'stress_33\']: ' + \
                                            '{:11.4e}'.format(state_variables['stress_33']))
                print('\n' + 'state_variables[\'is_su_fail\']:' + \
                                                         str(state_variables['is_su_fail']))
                print('\n' + 'consistent_tangent_mf:' + '\n')
                print(consistent_tangent_mf)
            # ------------------------------------------------------------------------------
            # Update cluster strain range indexes
            i_init = i_init + len(comp_order)
            i_end = i_init + len(comp_order)
    # Return
    return [clusters_state,clusters_D_mf]
# ------------------------------------------------------------------------------------------
# Update cluster interaction tensors and assemble global cluster interaction matrix
def updateCITs(problem_dict,material_properties_ref,Se_ref_matrix,material_phases,
          n_total_clusters,phase_n_clusters,phase_clusters,cit_1_mf,cit_2_mf,cit_0_freq_mf):
    # Get problem data
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
    Gop_factor_0_freq = numpy.matlib.repmat(Se_ref_matrix,n_total_clusters,n_total_clusters)
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
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[7]:
        print('\n' + 'global_cit_1_mf:' + '\n')
        print(global_cit_1_mf)
        print('\n' + 'global_cit_2_mf:' + '\n')
        print(global_cit_2_mf)
        print('\n' + 'global_cit_0_freq_mf:' + '\n')
        print(global_cit_0_freq_mf)
        print('\n' + 'Gop_factor_1: ' + str(Gop_factor_1))
        print('\n' + 'Gop_factor_2: ' + str(Gop_factor_2))
        print('\n' + 'Gop_factor_0_freq: ' + '\n')
        print(Gop_factor_0_freq)
        print('\n' + 'global_cit_mf: ' + '\n')
        print(global_cit_mf)
    # --------------------------------------------------------------------------------------
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
                    # ----------------------------------------------------------------------
                    # Validation:
                    if False:
                        section = 'assembleCIT'
                        print('\n' + '>>>> ' + section + ' ' + (92-len(section)-4)*'-')
                        print('\n' + 'mat_phase_A: ' + mat_phase_A + \
                                                      '  (clusterI: ' + str(clusterI) + ')')
                        print(       'mat_phase_B: ' + mat_phase_B + \
                                                      '  (clusterJ: ' + str(clusterJ) + ')')
                        print('\n' + 'cit_X_mf[mat_phase_pair][cluster_pair]:' + '\n')
                        print(cit_X_mf[mat_phase_pair][cluster_pair])
                    # ----------------------------------------------------------------------
                    # Set assembling ranges
                    i_init = iclst*len(comp_order)
                    i_end = i_init + len(comp_order)
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    # Assemble cluster interaction tensor
                    global_cit_X_mf[i_init:i_end,j_init:j_end] = \
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
                                 gbl_inc_strain_mf,global_cit_D_De_ref_mf,inc_mix_strain_mf,
                                                                   inc_mix_stress_mf,*args):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Initialize residual vector
    residual = np.zeros(n_total_clusters*len(comp_order) + n_presc_mac_stress)
    # Compute clusters equilibrium residuals
    residual[0:n_total_clusters*len(comp_order)] = gbl_inc_strain_mf + \
                                     np.matmul(global_cit_D_De_ref_mf,gbl_inc_strain_mf) - \
                                   numpy.matlib.repmat(inc_mix_strain_mf,1,n_total_clusters)
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
# Compute residuals of the discretized Lippmann-Schwinger system of nonlinear equilibrium
# equations (far-field strain implementation)
def buildResidual2(problem_dict,n_total_clusters,presc_strain_idxs,global_cit_D_De_ref_mf,
                      inc_hom_strain_mf,inc_hom_stress_mf,inc_mac_load_mf,gbl_inc_strain_mf,
                                                                    inc_farfield_strain_mf):
    # Get problem data
    comp_order = problem_dict['comp_order_sym']
    # Initialize residual vector
    residual = np.zeros(n_total_clusters*len(comp_order) + len(comp_order))
    # Compute clusters equilibrium residuals
    residual[0:n_total_clusters*len(comp_order)] = gbl_inc_strain_mf + \
                                     np.matmul(global_cit_D_De_ref_mf,gbl_inc_strain_mf) - \
                              numpy.matlib.repmat(inc_farfield_strain_mf,1,n_total_clusters)
    # Compute homogenization constraints residuals
    for i in range(len(comp_order)):
        if i in presc_strain_idxs:
            residual[n_total_clusters*len(comp_order) + i ] = \
                                         inc_hom_strain_mf[i] - inc_mac_load_mf['strain'][i]
        else:
            residual[n_total_clusters*len(comp_order) + i ] = \
                                         inc_hom_stress_mf[i] - inc_mac_load_mf['stress'][i]
    # Return
    return residual
# ------------------------------------------------------------------------------------------
# Compute Jacobian matrix of the discretized Lippmann-Schwinger system of nonlinear
# equilibrium equations
def buildJacobian(problem_dict,material_phases,phase_clusters,n_total_clusters,
      n_presc_mac_stress,presc_stress_idxs,global_cit_D_De_ref_mf,clusters_f,clusters_D_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set fourth-order identity tensor (matricial form)
    _,_,_,FOSym,_,_,_ = top.setIdentityTensors(n_dim)
    FOSym_mf = top.setTensorMatricialForm(FOSym,n_dim,comp_order)
    # Initialize Jacobian matrix
    Jacobian = np.zeros(2*(n_total_clusters*len(comp_order) + n_presc_mac_stress,))
    # Compute Jacobian matrix component solely related with the clusters equilibrium
    # residuals
    i_init = 0
    i_end = n_total_clusters*len(comp_order)
    j_init = 0
    j_end = n_total_clusters*len(comp_order)
    Jacobian[i_init:i_end,j_init:j_end] = \
                                scipy.linalg.block_diag(*(n_total_clusters*[FOSym_mf,])) + \
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
                  numpy.matlib.repmat(-1.0*FOSym_mf[:,presc_stress_idxs],n_total_clusters,1)
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
                    Jacobian[i,j_init:j_end] = f_D_mf[presc_stress_idxs[k],:]
                # Increment column cluster index
                jclst = jclst + 1
    # Return
    return Jacobian
# ------------------------------------------------------------------------------------------
# Compute Jacobian matrix of the discretized Lippmann-Schwinger system of nonlinear
# equilibrium equations
def buildJacobian2(problem_dict,material_phases,phase_clusters,n_total_clusters,
                         presc_strain_idxs,global_cit_D_De_ref_mf,clusters_f,clusters_D_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set fourth-order symmetric projection tensor (matricial form)
    _,_,_,FOSym,_,_,_ = top.setIdentityTensors(n_dim)
    FOSym_mf = top.setTensorMatricialForm(FOSym,n_dim,comp_order)
    # Initialize Jacobian matrix
    Jacobian = np.zeros(2*(n_total_clusters*len(comp_order) + len(comp_order),))
    # Compute Jacobian matrix component 11
    i_init = 0
    i_end = n_total_clusters*len(comp_order)
    j_init = 0
    j_end = n_total_clusters*len(comp_order)
    Jacobian[i_init:i_end,j_init:j_end] = \
                                scipy.linalg.block_diag(*(n_total_clusters*[FOSym_mf,])) + \
                                                                      global_cit_D_De_ref_mf
    # Compute Jacobian matrix component 12
    i_init = 0
    i_end = n_total_clusters*len(comp_order)
    j_init = n_total_clusters*len(comp_order)
    j_end = n_total_clusters*len(comp_order) + len(comp_order)
    Jacobian[i_init:i_end,j_init:j_end] = \
                                       numpy.matlib.repmat(-1.0*FOSym_mf,n_total_clusters,1)
    # Compute Jacobian matrix component 21
    for k in range(len(comp_order)):
        i = n_total_clusters*len(comp_order) + k
        jclst = 0
        for mat_phase in material_phases:
            for cluster in phase_clusters[mat_phase]:
                if k in presc_strain_idxs:
                    f_FOSym_mf = clusters_f[str(cluster)]*FOSym_mf
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    Jacobian[i,j_init:j_end] = f_FOSym_mf[k,:]
                else:
                    f_D_mf = clusters_f[str(cluster)]*clusters_D_mf[str(cluster)]
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    Jacobian[i,j_init:j_end] = f_D_mf[k,:]
                # Increment column cluster index
                jclst = jclst + 1
    # Return
    return Jacobian
# ------------------------------------------------------------------------------------------
# Compute homogenized strain and stress tensors (matricial form)
def homogenizedStrainStressTensors(problem_dict,material_phases,phase_clusters,clusters_f,
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
                                                               inc_strain_mf,inc_stress_mf):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Set second-order identity tensor
    SOId,_,_,_,_,_,_ = top.setIdentityTensors(n_dim)
    # Perform self-consistent scheme to update the reference material elastic properties
    # 1. Regression-based scheme
    # 2. Projection-based scheme
    if self_consistent_scheme == 1:
        # Initialize self-consistent scheme system of linear equations coefficient matrix
        # and right-hand side
        scs_matrix = np.zeros((2,2))
        scs_rhs = np.zeros(2)
        # Get incremental strain and stress tensors
        inc_strain = top.getTensorFromMatricialForm(inc_strain_mf,n_dim,comp_order)
        inc_stress = top.getTensorFromMatricialForm(inc_stress_mf,n_dim,comp_order)
        # Compute self-consistent scheme system of linear equations right-hand side
        scs_rhs[0] = np.trace(inc_stress)
        scs_rhs[1] = top.ddot22_1(inc_stress,inc_strain)
        # Compute self-consistent scheme system of linear equations coefficient matrix
        scs_matrix[0,0] = np.trace(inc_strain)*np.trace(SOId)
        scs_matrix[0,1] = 2.0*np.trace(inc_strain)
        scs_matrix[1,0] = np.trace(inc_strain)**2
        scs_matrix[1,1] = 2.0*top.ddot22_1(inc_strain,inc_strain)
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
# ------------------------------------------------------------------------------------------
# Check Newton-Raphson iterative procedure convergence when solving the Lippmann-Schwinger
# nonlinear system of equilibrium equations associated to a given macroscale load
# increment
def checkNRConvergence(comp_order,mac_load_presctype,n_total_clusters,inc_mac_load_mf,
                   n_presc_mac_stress,inc_hom_strain_mf,inc_hom_stress_mf,inc_mix_strain_mf,
                                                       inc_mix_stress_mf,residual,conv_tol):
    # Initialize criterion convergence flag
    is_converged_A = True
    # Loop over strain/stress components
    for i in range(len(comp_order)):
        comp = comp_order[i]
        # If the prescribed incremental macroscale component is null, then skip its
        # evaluation
        if inc_mac_load_mf[mac_load_presctype[comp]][i] < 1e-6:
            continue
        # Compute error to check if the incremental homogenized strain/stress component
        # converged to the prescribed macroscale value
        if mac_load_presctype[comp] == 'strain':
            error_A = abs((inc_hom_strain_mf[i] - inc_mac_load_mf['strain'][i])/ \
                                                               inc_mac_load_mf['strain'][i])
        else:
            error_A = abs((inc_hom_stress_mf[i] - inc_mac_load_mf['stress'][i])/ \
                                                               inc_mac_load_mf['stress'][i])
        # If at least one of the strain/stress components did not converge according to the
        # defined convergence tolerance, then switch the criterion convergence flag to False
        # and leave criterion evaluation
        if error_A > conv_tol:
            is_converged_A = False
            break
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute error associated to the clusters equilibrium residuals
    error_B1 = np.linalg.norm(residual[0:n_total_clusters*len(comp_order)])/ \
                                                           np.linalg.norm(inc_mix_strain_mf)
    # Compute error associated to the prescribed macroscale stress components residuals
    error_B2 = None
    if n_presc_mac_stress > 0:
        error_B2 = np.linalg.norm(residual[n_total_clusters*len(comp_order):])/ \
                                                           np.linalg.norm(inc_mix_stress_mf)
    # Criterion convergence flag is True if both residual errors converged according to the
    # defined convergence tolerance
    if n_presc_mac_stress > 0:
        is_converged_B = (error_B1 < conv_tol) and (error_B2 < conv_tol)
    else:
        is_converged_B = error_B1 < conv_tol
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[14]:
        section = 'Convergence evaluation'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'error_A (at break) = ' + str(error_A))
        print('\n' + 'error_B1 = ' + str(error_B1))
        print('\n' + 'error_B2 = ' + str(error_B2))
        print('\n' + 'is_converged_A = ' + str(is_converged_A))
        print('\n' + 'is_converged_B = ' + str(is_converged_B))
    # --------------------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [is_converged_A,is_converged_B,error_B1,error_B2]
# ------------------------------------------------------------------------------------------
# Check Newton-Raphson iterative procedure convergence when solving the Lippmann-Schwinger
# nonlinear system of equilibrium equations associated to a given macroscale load
# increment
def checkNRConvergence2(comp_order,n_total_clusters,inc_mac_load_mf,n_presc_mac_strain,
                                     n_presc_mac_stress,presc_strain_idxs,presc_stress_idxs,
                                                       inc_hom_strain_mf,residual,conv_tol):
    # Initialize criterion convergence flag
    is_converged = False
    # Set strain and stress normalization factors
    if n_presc_mac_strain > 0:
        strain_norm_factor = np.linalg.norm(inc_mac_load_mf['strain'][[presc_strain_idxs]])
    elif not np.allclose(inc_hom_strain_mf,np.zeros(inc_hom_strain_mf.shape),atol=1e-10):
        strain_norm_factor = np.linalg.norm(inc_hom_strain_mf)
    else:
        strain_norm_factor = 1
    if n_presc_mac_stress > 0:
        stress_norm_factor = np.linalg.norm(inc_mac_load_mf['stress'][[presc_stress_idxs]])
    # Compute error associated to the clusters equilibrium residuals
    error_A1 = np.linalg.norm(residual[0:n_total_clusters*len(comp_order)])/ \
                                                                          strain_norm_factor
    # Compute error associated to the homogenization constraints residuals
    aux = residual[n_total_clusters*len(comp_order):]
    if n_presc_mac_strain > 0:
        error_A2 = np.linalg.norm(aux[presc_strain_idxs])/strain_norm_factor
    if n_presc_mac_stress > 0:
        error_A3 = np.linalg.norm(aux[presc_stress_idxs])/stress_norm_factor

    # Criterion convergence flag is True if all residual errors converged according to the
    # defined convergence tolerance
    if n_presc_mac_strain == 0:
        error_A2 = None
        is_converged = (error_A1 < conv_tol) and (error_A3 < conv_tol)
    elif n_presc_mac_stress == 0:
        error_A3 = None
        is_converged = (error_A1 < conv_tol) and (error_A2 < conv_tol)
    else:
        is_converged = (error_A1 < conv_tol) and (error_A2 < conv_tol) \
                                                                   and (error_A3 < conv_tol)
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[14]:
        section = 'Convergence evaluation'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'error_A1 = ' + str(error_A1))
        print('\n' + 'error_A2 = ' + str(error_A2))
        print('\n' + 'error_A3 = ' + str(error_A3))
        print('\n' + 'is_converged = ' + str(is_converged))
        print('\n' + 'conv_tol = ' + str(conv_tol))
    # --------------------------------------------------------------------------------------
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [is_converged,error_A1,error_A2,error_A3]
# ------------------------------------------------------------------------------------------
# Check self-consistent scheme iterative procedure convergence
def checkSCSConvergence(E_ref,v_ref,material_properties_ref,scs_conv_tol):
    # Compute iterative variation of the reference material Young modulus and Poisson ratio
    d_E_ref = E_ref - material_properties_ref['E']
    d_v_ref = v_ref - material_properties_ref['v']
    # Compute normalized interative change of the reference material Young modulus and
    # Poisson ratio
    norm_d_E_ref = abs(d_E_ref/E_ref)
    norm_d_v_ref = abs(d_v_ref/v_ref)
    # The self-consistent scheme convergence flags are True if the normalized iterative
    # change of the reference material elastic property converged according to the defined
    # convergence tolerance
    is_scs_conv_E = norm_d_E_ref < scs_conv_tol
    is_scs_conv_v = norm_d_v_ref < scs_conv_tol
    # ------------------------------------------------------------------------------
    # Validation:
    if is_Validation[20]:
        section = 'Self-consistent scheme convergence evaluation'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'd_E_ref = ' + '{:11.4e}'.format(d_E_ref))
        print('\n' + 'd_v_ref = ' + '{:11.4e}'.format(d_v_ref))
        print('\n' + 'norm_d_E_ref = ' + '{:11.4e}'.format(norm_d_E_ref))
        print('\n' + 'norm_d_v_ref = ' + '{:11.4e}'.format(norm_d_v_ref))
    # ------------------------------------------------------------------------------
    # Return
    return [is_scs_conv_E,is_scs_conv_v,norm_d_E_ref,norm_d_v_ref]
