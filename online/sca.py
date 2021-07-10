#
# SCA Online Stage Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing the solution procedure of the Lippmann-Schwinger nonlinear system of
# equilibrium equations embedded in a self-consistent scheme, the core of the
# Self-Consistent Clustering Analysis (SCA) online stage.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Dec 2020 | Implemented clustering adaptivity.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
import numpy.matlib
# Date and time
import time
# Shallow and deep copy operations
import copy
# Display messages
import ioput.info as info
# Scientific computation
import scipy.linalg
# Matricial operations
import tensor.matrixoperations as mop
# Cluster interaction tensors operations
import clustering.citoperations as citop
# Homogenized results output
from ioput.homresoutput import HomResOutput
# Reference material output
from ioput.refmatoutput import RefMatOutput
# VTK output
from ioput.vtkoutput import VTKOutput
# Material interface
import material.materialinterface
# Linear elastic constitutive model
import material.models.linear_elastic
# Macroscale load incrementation
from online.incrementation.macloadincrem import LoadingPath
# Homogenization
import online.homogenization.homogenization as hom
# Clusters state
import online.clusters.clusters_suct as clstsuct
# Self-consistent scheme
import online.scs.scs_schemes as scs
# Lippmann-Schwinger nonlinear system of equilibrium equations
import online.equilibrium.sne_farfield as eqff
import online.equilibrium.sne_macstrain as eqms
# CRVE adaptivity
from clustering.adaptivity.crve_adaptivity import AdaptivityManager, \
                                                  ClusteringAdaptivityOutput
# Voxels output
from ioput.voxelsoutput import VoxelsOutput



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
#  8. Global residual matrix 1 (Zeliang approach) (removed)
#  9. Clusters material state update and consistent tangent
# 10. Global residual matrix 2 (Zeliang approach) (removed)
# 11. Global cluster interaction - tangent matrix
# 12. Incremental homogenized strain and stress tensors
# 13. Lippmann-Schwinger SNLE residuals
# 14. Convergence evaluation
# 15. Lippmann-Schwinger SNLE Jacobian
# 16. Lippmann-Schwinger SLE solution
# 17. Incremental strains iterative update
# 18. Incremental homogenized strain and stress tensors (removed)
# 19. Self-consistent scheme
# 20. Self-consistent scheme convergence evaluation
# 21. Homogenized strain and stress tensors
#output_idx = [0,12,14,18,19]
output_idx = []
is_Validation = [False for i in range(22)]
for i in output_idx:
    is_Validation[i] = True
#
#                                             Solution of the discretized Lippmann-Schwinger
#                                                  system of nonlinear equilibrium equations
# ==========================================================================================
def sca(dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict, macload_dict, scs_dict,
        algpar_dict, vtk_dict, output_dict, crve):
    #                                                                           General data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get input data file name and post processing directory
    input_file_name = dirs_dict['input_file_name']
    postprocess_dir = dirs_dict['postprocess_dir']
    hres_file_path = dirs_dict['hres_file_path']
    refm_file_path = dirs_dict['refm_file_path']
    adapt_file_path = dirs_dict['adapt_file_path']
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    strain_formulation = problem_dict['strain_formulation']
    # Get material data
    material_phases = mat_dict['material_phases']
    material_phases_f = mat_dict['material_phases_f']
    material_properties = mat_dict['material_properties']
    material_phases_models = mat_dict['material_phases_models']
    # Get clusters data
    phase_n_clusters = clst_dict['phase_n_clusters']
    phase_clusters = clst_dict['phase_clusters']
    clusters_f = clst_dict['clusters_f']
    # Get macroscale loading data
    mac_load = macload_dict['mac_load']
    mac_load_presctype = macload_dict['mac_load_presctype']
    mac_load_increm = macload_dict['mac_load_increm']
    # Get self-consistent scheme data
    self_consistent_scheme = scs_dict['self_consistent_scheme']
    scs_max_n_iterations = scs_dict['scs_max_n_iterations']
    scs_conv_tol = scs_dict['scs_conv_tol']
    # Get algorithmic parameters
    max_n_iterations = algpar_dict['max_n_iterations']
    conv_tol = algpar_dict['conv_tol']
    max_subinc_level = algpar_dict['max_subinc_level']
    max_cinc_cuts = algpar_dict['max_cinc_cuts']
    # Get VTK output parameters
    is_VTK_output = vtk_dict['is_VTK_output']
    if is_VTK_output:
        # Set VTK output files parameters
        vtk_dir = postprocess_dir + 'VTK/'
        pvd_dir = postprocess_dir
        vtk_byte_order = vtk_dict['vtk_byte_order']
        vtk_format = vtk_dict['vtk_format']
        vtk_precision = vtk_dict['vtk_precision']
        vtk_vars = vtk_dict['vtk_vars']
        vtk_inc_div = vtk_dict['vtk_inc_div']
        # Instantiante VTK output
        vtk_output = VTKOutput(type='ImageData', version='1.0', byte_order=vtk_byte_order,
                               format=vtk_format, precision=vtk_precision,
                               header_type='UInt64', base_name=input_file_name,
                               vtk_dir=vtk_dir, pvd_dir=pvd_dir)
    # Get general output parameters
    is_voxels_output = output_dict['is_voxels_output']
    if is_voxels_output:
        voxout_file_path = dirs_dict['voxout_file_path']
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
    ons_init_time = time.time()
    # Initialize online stage post-processing time
    ons_post_process_time = 0.0
    # Set far-field formulation flag
    is_farfield_formulation = True
    # Initialize homogenized strain and stress tensors
    hom_strain_mf = np.zeros(len(comp_order))
    hom_strain_old_mf = np.zeros(len(comp_order))
    hom_stress_mf = np.zeros(len(comp_order))
    hom_stress_old_mf = np.zeros(len(comp_order))
    if problem_type == 1:
        hom_stress_33 = 0.0
        hom_stress_33_old = 0.0
    # Initialize clusters state variables dictionaries
    clusters_state = dict()
    clusters_state_old = dict()
    # Initialize clusters state variables
    for mat_phase in material_phases:
        # Loop over material phase clusters
        for cluster in phase_clusters[mat_phase]:
            # Initialize state variables
            clusters_state[str(cluster)] = \
                material.materialinterface.materialinterface('init', problem_dict, mat_dict,
                                                             clst_dict, algpar_dict,
                                                             mat_phase)
            clusters_state_old[str(cluster)] = \
                material.materialinterface.materialinterface('init', problem_dict, mat_dict,
                                                             clst_dict, algpar_dict,
                                                             mat_phase)
    # Initialize clusters strain concentration tensor dictionary
    clusters_sct_mf_old = \
        hom.init_clusters_sct(n_dim, comp_order, material_phases, phase_clusters)
    # Get total number of clusters
    n_total_clusters = crve.get_n_total_clusters()
    # Initialize macroscale loading increment cut flag
    is_inc_cut = False
    # Initialize flag that locks reference material elastic properties
    is_lock_prop_ref = False
    # Initialize flag that signals an improved cluster incremental strains initial iterative
    # guess
    is_improved_init_guess = False
    # Check clustering adaptivity and perform required initializations
    is_crve_adaptivity = False
    adaptivity_manager = None
    if len(crve.adapt_material_phases) > 0:
        # Switch on clustering adaptivity flag
        is_crve_adaptivity = True
        # Set flag that controls if the macroscale loading increment where the clustering
        # adaptivity is triggered is to be repeated considering the new clustering
        is_adapt_repeat_inc = True
        # Get clustering adaptivity frequency
        clust_adapt_freq = clst_dict['clust_adapt_freq']
        # Get clustering adaptivity output
        is_clust_adapt_output = clst_dict['is_clust_adapt_output']
        # Initialize online CRVE clustering adaptivity manager
        adaptivity_manager = \
            AdaptivityManager(problem_type, comp_order, crve.adapt_material_phases,
                              crve.phase_clusters, crve.adaptivity_control_feature,
                              crve.adapt_criterion_data, clust_adapt_freq)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set post-processing procedure initial time
        procedure_init_time = time.time()
        # Instantiate clustering adaptivity output
        adapt_output = ClusteringAdaptivityOutput(adapt_file_path,
                                                  crve.adapt_material_phases)
        # Write clustering adaptivity output file header
        adapt_output.write_adapt_file(0, adaptivity_manager, crve, mode='init')
        # Increment post-processing time
        ons_post_process_time += time.time() - procedure_init_time
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[1]:
        section = 'Initializations'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'n_total_clusters = ' + str(n_total_clusters))
    # --------------------------------------------------------------------------------------
    #
    #                                         Initial state homogenized results (.hres file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build homogenized strain tensor
    hom_strain = mop.gettensorfrommf(hom_strain_mf, n_dim, comp_order)
    # Build homogenized stress tensor
    hom_stress = mop.gettensorfrommf(hom_stress_mf, n_dim, comp_order)
    # Compute homogenized out-of-plane stress component in a 2D plane strain problem /
    # strain component in a 2D plane stress problem (output purpose only)
    if problem_type == 1:
        hom_stress_33 = hom.homoutofplanecomp(problem_type, material_phases, phase_clusters,
                                              clusters_f, clusters_state)
    # Initialize homogenized results dictionary
    hom_results = dict()
    # Build homogenized results dictionary
    hom_results['hom_strain'] = hom_strain
    hom_results['hom_stress'] = hom_stress
    if problem_type == 1:
        hom_results['hom_stress_33'] = hom_stress_33
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set post-processing procedure initial time
    procedure_init_time = time.time()
    # Instantiate homogenized results output
    hres_output = HomResOutput(hres_file_path)
    # Write homogenized results output file header
    hres_output.init_hres_file()
    # Write homogenized results initial state
    hres_output.write_hres_file(problem_type, 0, hom_results)
    # Increment post-processing time
    ons_post_process_time += time.time() - procedure_init_time
    #
    #                                            Reference material output file (.refm file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set post-processing procedure initial time
    procedure_init_time = time.time()
    # Instantiate reference material output
    ref_mat_output = RefMatOutput(refm_file_path, self_consistent_scheme,
                                  is_farfield_formulation)
    # Write reference material output file header
    ref_mat_output.init_ref_mat_file()
    # Increment post-processing time
    ons_post_process_time += time.time() - procedure_init_time
    #
    #                          Voxels material-related quantities output file (.voxout file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_voxels_output:
        # Set post-processing procedure initial time
        procedure_init_time = time.time()
        # Instantiate voxels material-related output
        voxels_output = VoxelsOutput(voxout_file_path, problem_type)
        # Write reference material output file header
        voxels_output.init_voxels_output_file(crve)
        # Increment post-processing time
        ons_post_process_time += time.time() - procedure_init_time
    #
    #                                                                 Initial state VTK file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_VTK_output:
        # Set post-processing procedure initial time
        procedure_init_time = time.time()
        # Write VTK file associated to the initial state
        vtk_output.write_VTK_file_time_step(time_step=0,
            problem_type=problem_type, crve=crve, clusters_state=clusters_state,
                material_phases_models=material_phases_models, vtk_vars=vtk_vars,
                    adaptivity_manager=adaptivity_manager)
        # Increment post-processing time
        ons_post_process_time += time.time() - procedure_init_time
    #
    #                                                      Material clusters elastic tangent
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elastic tangent (matricial form) associated to each material cluster
    clusters_De_mf = clstsuct.clusterselastictanmod(problem_dict, material_properties,
                                                    material_phases, phase_clusters)
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
    #                                                     Reference material elastic tangent
    #                                                                        (initial guess)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set reference material elastic properties initial guess based on the volume averages
    # of the material phases elastic properties
    E_ref = sum([material_phases_f[phase]*material_properties[phase]['E']
        for phase in material_phases])
    v_ref = sum([material_phases_f[phase]*material_properties[phase]['v']
        for phase in material_phases])
    mat_prop_ref = dict()
    mat_prop_ref['E'] = E_ref
    mat_prop_ref['v'] = v_ref
    mat_prop_ref_old = copy.deepcopy(mat_prop_ref)
    # Compute the reference material elastic tangent (matricial form) and compliance tensor
    # (matrix)
    De_ref_mf, Se_ref_matrix = scs.refelastictanmod(problem_dict, mat_prop_ref)
    # Initialize reference material elastic properties from first macroscale loading
    # increment
    mat_prop_ref_first = dict.fromkeys(mat_prop_ref.keys())
    # --------------------------------------------------------------------------------------
    # Validation:
    if is_Validation[4]:
        section = 'Reference material elastic tangent'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'mat_prop_ref[\'E\'] = '+ str(mat_prop_ref['E']))
        print('\n' + 'mat_prop_ref[\'v\'] = '+ str(mat_prop_ref['v']))
        print('\n' + 'De_ref_mf:' + '\n')
        print(De_ref_mf)
        print('\n' + 'Se_ref_matrix:' + '\n')
        print(Se_ref_matrix)
    # --------------------------------------------------------------------------------------
    #
    #                                                                Macroscale loading path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize macroscale loading path
    mac_load_path = LoadingPath(strain_formulation, problem_dict['comp_order_sym'],
                                problem_dict['comp_order_nsym'], mac_load,
                                mac_load_presctype, mac_load_increm, max_subinc_level,
                                max_cinc_cuts)
    # Set initial homogenized state
    mac_load_path.update_hom_state(n_dim, comp_order, hom_strain, hom_stress)
    # Setup first macroscale loading increment
    inc_mac_load_mf, n_presc_strain, presc_strain_idxs, n_presc_stress, \
        presc_stress_idxs, is_last_inc = mac_load_path.new_load_increment(n_dim, comp_order)
    # Get increment counter
    inc = mac_load_path.increm_state['inc']
    # Display increment data
    displayincdata(mac_load_path)
    # Set increment initial time
    inc_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Validation:
    if any(is_Validation):
        #print('\n' + (92 - len(' Increment ' + str(inc)))*'-' + ' Increment ' + str(inc))
        section = 'Macroscale load increments'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        if n_presc_strain > 0:
            print('\n' + 'n_presc_strain = ' + str(n_presc_strain))
            print('\n' + 'presc_strain_idxs = ' + str(presc_strain_idxs))
            print('\n' + 'inc_mac_load_mf[strain]:' + '\n')
            print(inc_mac_load_mf['strain'])
        if n_presc_stress > 0:
            print('\n' + 'n_presc_stress = ' + str(n_presc_stress))
            print('\n' + 'presc_stress_idxs = ' + str(presc_stress_idxs))
            print('\n' + 'inc_mac_load_mf[stress]:' + '\n')
            print(inc_mac_load_mf['stress'])
    # --------------------------------------------------------------------------------------
    # Start incremental loading loop
    while True:
        #                                                        Incremental macroscale load
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not is_farfield_formulation:
            # Initialize strain tensor where each component is defined as follows:
            # (a) Incremental macroscale strain (if prescribed macroscale strain component)
            # (b) Incremental homogenized strain (if non-prescribed macroscale strain
            # component)
            inc_mix_strain_mf = np.zeros(len(comp_order))
            if n_presc_strain > 0:
                inc_mix_strain_mf[presc_strain_idxs] = \
                    inc_mac_load_mf['strain'][presc_strain_idxs]
            # Initialize stress tensor where each component is defined as follows:
            # (a) Incremental macroscale stress (if prescribed macroscale stress component)
            # (b) Incremental homogenized stress (if non-prescribed macroscale stress
            #     component)
            inc_mix_stress_mf = np.zeros(len(comp_order))
            if n_presc_stress > 0:
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
        # Set clusters incremental strain initial iterative guess
        if not is_improved_init_guess:
            gbl_inc_strain_mf = np.zeros((n_total_clusters*len(comp_order)))
        else:
            is_improved_init_guess = False
        # Set additional initial iterative guesses
        if is_farfield_formulation:
            # Set incremental far-field strain initial iterative guess
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
            if is_farfield_formulation:
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
        if is_lock_prop_ref:
            info.displayinfo('13', 'init', scs_iter, mat_prop_ref['E'], mat_prop_ref['v'])
        else:
            info.displayinfo('8', 'init', scs_iter, mat_prop_ref['E'], mat_prop_ref['v'])
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
            global_cit_mf = citop.updassemblecit(
                problem_dict, mat_prop_ref, Se_ref_matrix, material_phases,
                phase_n_clusters, phase_clusters, crve.cit_X_mf[0], crve.cit_X_mf[1],
                crve.cit_X_mf[2])
            #
            #                                                  Newton-Raphson iterative loop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize Newton-Raphson iteration counter
            nr_iter = 0
            if is_farfield_formulation:
                info.displayinfo('9', 'init')
            else:
                info.displayinfo('10', 'init')
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
                clusters_state, clusters_D_mf, su_fail_state = clstsuct.clusterssuct(
                    problem_dict, mat_dict, clst_dict, algpar_dict, phase_clusters,
                    gbl_inc_strain_mf, clusters_state_old)
                # Raise macroscale increment cut procedure if material cluster state update
                # failed
                if su_fail_state['is_su_fail']:
                    is_inc_cut = True
                    # Display increment cut
                    info.displayinfo('11', 'su_fail', su_fail_state)
                    # Leave Newton-Raphson equilibrium iterative loop
                    break
                #
                #                                Global cluster interaction - tangent matrix
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
                global_cit_D_De_ref_mf = np.matmul(global_cit_mf,
                    scipy.linalg.block_diag(*diff_D_De_ref_mf))
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[11]:
                    section = 'Global interaction Jacobian matrix'
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
                hom_strain_mf, hom_stress_mf = \
                    hom.homstatetensors(comp_order, material_phases, phase_clusters,
                                        clusters_f, clusters_state)
                # Compute homogenized out-of-plane stress component in a 2D plane strain
                # problem / strain component in a 2D plane stress problem
                if problem_type == 1:
                    hom_stress_33 = hom.homoutofplanecomp(
                        problem_type, material_phases, phase_clusters, clusters_f,
                        clusters_state)
                # Compute incremental homogenized strain and stress tensors (matricial form)
                inc_hom_strain_mf = hom_strain_mf - hom_strain_old_mf
                inc_hom_stress_mf = hom_stress_mf - hom_stress_old_mf
                if problem_type == 1:
                    inc_hom_stress_33 = hom_stress_33 - hom_stress_33_old
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
                    if is_farfield_formulation:
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
                if is_farfield_formulation:
                    residual = eqff.buildresidual2(
                        problem_dict, material_phases, phase_clusters, n_total_clusters,
                        presc_strain_idxs, global_cit_mf, clusters_state,
                        clusters_state_old, De_ref_mf, inc_hom_strain_mf, inc_hom_stress_mf,
                        inc_mac_load_mf, gbl_inc_strain_mf, inc_farfield_strain_mf)
                else:
                    residual = eqms.buildresidual(
                        problem_dict, material_phases, phase_clusters, n_total_clusters,
                        n_presc_stress, presc_stress_idxs, global_cit_mf,
                        clusters_state, clusters_state_old, De_ref_mf,
                        gbl_inc_strain_mf, inc_mix_strain_mf,
                        inc_hom_stress_mf, inc_mix_stress_mf)
                # --------------------------------------------------------------------------
                # Validation:
                if is_Validation[13]:
                    section = 'Lippmann-Schwinger SNLE residuals'
                    print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                    print('\n' + 'residual:' + '\n')
                    #print(residual)
                    for i in range(len(residual)):
                        print('{:13.4e}'.format(residual[i]))
                        if i != 0 and (i + 1) % 6 == 0:
                            print('')
                # --------------------------------------------------------------------------
                #
                #                                                     Convergence evaluation
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check Newton-Raphson iterative procedure convergence
                if is_farfield_formulation:
                    is_converged, error_A1, error_A2, error_A3 = eqff.checkeqlbconvergence2(
                        comp_order, n_total_clusters, inc_mac_load_mf, n_presc_strain,
                        n_presc_stress, presc_strain_idxs, presc_stress_idxs,
                        inc_hom_strain_mf, inc_hom_stress_mf, residual, conv_tol)
                    info.displayinfo('9', 'iter', nr_iter, time.time() - nr_iter_init_time,
                                     error_A1, error_A2, error_A3)
                else:
                    is_converged, error_A1, error_A2, error_inc_hom_strain = \
                        eqms.checkeqlbconvergence(
                            comp_order, n_total_clusters, inc_mac_load_mf,
                            n_presc_strain, n_presc_stress, presc_strain_idxs,
                            presc_stress_idxs, inc_hom_strain_mf, inc_hom_stress_mf,
                            inc_mix_strain_mf, inc_mix_stress_mf, residual, conv_tol)
                    info.displayinfo('10', 'iter', nr_iter, time.time() - nr_iter_init_time,
                                     error_A1, error_A2, error_inc_hom_strain)
                # Control Newton-Raphson iteration loop flow
                if is_converged:
                    # Leave Newton-Raphson iterative loop (converged solution)
                    break
                elif nr_iter == max_n_iterations:
                    # Raise macroscale increment cut procedure
                    is_inc_cut = True
                    # Display increment cut
                    info.displayinfo('11', 'max_iter', max_n_iterations)
                    # Leave Newton-Raphson equilibrium iterative loop
                    break
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
                if is_farfield_formulation:
                    Jacobian = eqff.buildjacobian2(
                        problem_dict, material_phases, phase_clusters, n_total_clusters,
                        presc_strain_idxs, global_cit_D_De_ref_mf, clusters_f,
                        clusters_D_mf)
                else:
                    Jacobian = eqms.buildjacobian(
                        problem_dict, material_phases, phase_clusters, n_total_clusters,
                        n_presc_stress, presc_stress_idxs, global_cit_D_De_ref_mf,
                        clusters_f, clusters_D_mf)
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
                d_iter = numpy.linalg.solve(Jacobian, -residual)
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
                if is_farfield_formulation:
                    # Update far-field strain
                    inc_farfield_strain_mf = inc_farfield_strain_mf + \
                        d_iter[n_total_clusters*len(comp_order):]
                else:
                    # Update homogenized incremental strain components
                    if n_presc_stress > 0:
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
                    if is_farfield_formulation:
                        print('\n' + 'inc_farfield_strain_mf:' + '\n')
                        print(inc_farfield_strain_mf)
                    else:
                        if n_presc_stress > 0:
                            print('\n' + 'inc_mix_strain_mf:' + '\n')
                            print(inc_mix_strain_mf)
                # --------------------------------------------------------------------------
            #
            #                                                         Self-consistent scheme
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If raising a macroscale loading increment cut, leave self-consistent iterative
            # loop
            if is_inc_cut:
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute CRVE effective tangent modulus and clusters strain concentration
            # tensors
            eff_tangent_mf, clusters_sct_mf = \
                hom.effective_tangent_modulus(n_dim, comp_order, material_phases,
                                              phase_clusters, clusters_f, clusters_D_mf,
                                              global_cit_D_De_ref_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set post-processing procedure initial time
            procedure_init_time = time.time()
            # Output reference material associated quantities (.refm file)
            if is_farfield_formulation:
                ref_mat_output.write_ref_mat(problem_type, n_dim, comp_order, inc, scs_iter,
                                             mat_prop_ref, De_ref_mf,
                                             inc_hom_strain_mf, inc_hom_stress_mf,
                                             eff_tangent_mf, inc_farfield_strain_mf,
                                             inc_mac_load_mf['strain'])
            else:
                ref_mat_output.write_ref_mat(problem_type, n_dim, comp_order, inc, scs_iter,
                                             mat_prop_ref, De_ref_mf,
                                             inc_hom_strain_mf, inc_hom_stress_mf,
                                             eff_tangent_mf)
            # Increment post-processing time
            ons_post_process_time += time.time() - procedure_init_time
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update reference material elastic properties through a given self-consistent
            # scheme
            if is_lock_prop_ref:
                # Skip update of reference material elastic properties
                E_ref = mat_prop_ref['E']
                v_ref = mat_prop_ref['v']
            else:
                # Set self-consistent scheme arguments
                if self_consistent_scheme == 1:
                    scs_args = (self_consistent_scheme, problem_dict, inc_hom_strain_mf,
                                inc_hom_stress_mf, mat_prop_ref)
                    if problem_type == 1:
                        scs_args = scs_args + (inc_hom_stress_33,)
                elif self_consistent_scheme == 2:
                    if is_farfield_formulation:
                        scs_args = (self_consistent_scheme, problem_dict,
                                    inc_farfield_strain_mf, inc_hom_stress_mf, mat_prop_ref,
                                    eff_tangent_mf)
                    else:
                        scs_args = (self_consistent_scheme, problem_dict, inc_hom_strain_mf,
                                    inc_hom_stress_mf, mat_prop_ref, eff_tangent_mf)
                # Update reference material elastic properties
                E_ref, v_ref = scs.scsupdate(*scs_args)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Evaluate admissibility of self-consistent scheme iterative solution
                if inc == 1:
                    is_scs_admissible = True
                else:
                    is_scs_admissible = scs.check_scs_solution(E_ref, v_ref,
                                                               mat_prop_ref_first)
                # If self-consistent scheme iterative solution is not admissible, either
                # accept the current solution (first self-consistent scheme iteration) or
                # perform one last self-consistent scheme iteration with the last converged
                # increment reference material elastic properties
                if not is_scs_admissible:
                    info.displayinfo('8', 'end', time.time() - scs_iter_init_time)
                    if scs_iter == 0:
                        # Display locking of reference material elastic properties
                        info.displayinfo('14', 'locked_scs_solution')
                        # Leave self-consistent scheme iterative loop (accepted solution)
                        break
                    else:
                        # Display locking of reference material elastic properties
                        info.displayinfo('14', 'inadmissible_scs_solution')
                        # If inadmissible self-consistent scheme solution, reset reference
                        # material elastic properties to the last converged increment values
                        mat_prop_ref = copy.deepcopy(mat_prop_ref_old)
                        # Lock reference material elastic properties
                        is_lock_prop_ref = True
                        # Perform one last self-consistent scheme iteration with the last
                        # converged increment reference material elastic properties
                        info.displayinfo('13', 'init', 0, mat_prop_ref['E'],
                                         mat_prop_ref['v'])
                        # Compute the reference material elastic tangent (matricial form)
                        # and compliance tensor (matrix)
                        De_ref_mf, Se_ref_matrix = scs.refelastictanmod(problem_dict,
                                                                        mat_prop_ref)
                        # Proceed to last self-consistent scheme iteration
                        continue
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
            is_scs_converged, norm_d_E_ref, norm_d_v_ref = \
                scs.checkscsconvergence(E_ref, v_ref, mat_prop_ref, scs_conv_tol)
            info.displayinfo('8', 'end', time.time() - scs_iter_init_time)
            # Control self-consistent scheme iteration loop flow
            if is_scs_converged:
                # Reset flag that locks reference material elastic properties
                is_lock_prop_ref = False
                # Leave self-consistent scheme iterative loop (converged solution)
                break
            elif scs_iter == scs_max_n_iterations:
                # Display locking of reference material elastic properties
                info.displayinfo('14', 'max_scs_iter', scs_max_n_iterations)
                # If the maximum number of self-consistent scheme iterations is reached
                # without convergence, reset reference material elastic properties to the
                # last converged increment values
                mat_prop_ref = copy.deepcopy(mat_prop_ref_old)
                # Lock reference material elastic properties
                is_lock_prop_ref = True
                # Perform one last self-consistent scheme iteration with the last converged
                # increment reference material elastic properties
                info.displayinfo('13', 'init', 0, mat_prop_ref['E'], mat_prop_ref['v'])
            else:
                # Update reference material elastic properties
                mat_prop_ref['E'] = E_ref
                mat_prop_ref['v'] = v_ref
                # Increment self-consistent scheme iteration counter
                scs_iter = scs_iter + 1
                info.displayinfo('8', 'init', scs_iter, E_ref, norm_d_E_ref, v_ref,
                                 norm_d_v_ref)
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
            De_ref_mf, Se_ref_matrix = scs.refelastictanmod(problem_dict, mat_prop_ref)
            # ------------------------------------------------------------------------------
            # Validation:
            if is_Validation[4]:
                section = 'Reference material elastic tangent'
                print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
                print('\n' + 'mat_prop_ref[\'E\'] = ' + str(mat_prop_ref['E']))
                print('\n' + 'mat_prop_ref[\'v\'] = ' + str(mat_prop_ref['v']))
                print('\n' + 'De_ref_mf:' + '\n')
                print(De_ref_mf)
                print('\n' + 'Se_ref_matrix:' + '\n')
                print(Se_ref_matrix)
            # ------------------------------------------------------------------------------
        #
        #                                                   Macroscale loading increment cut
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_inc_cut:
            # Reset macroscale loading increment cut flag
            is_inc_cut = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Reset reference material elastic properties to the last converged increment
            # values and leave self-consistent iterative loop
            mat_prop_ref = copy.deepcopy(mat_prop_ref_old)
            # Compute the reference material elastic tangent (matricial form) and compliance
            # tensor (matrix)
            De_ref_mf, Se_ref_matrix = scs.refelastictanmod(problem_dict, mat_prop_ref)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform macroscale increment cut and setup new macroscale loading increment
            inc_mac_load_mf, n_presc_strain, presc_strain_idxs, n_presc_stress, \
                presc_stress_idxs, is_last_inc = mac_load_path.increment_cut(
                    n_dim, comp_order)
            # Get increment counter
            inc = mac_load_path.increm_state['inc']
            # Display increment data
            displayincdata(mac_load_path)
            # Set increment initial time
            inc_init_time = time.time()
            # Start new macroscale loading increment solution procedures
            continue
        #
        #                                                              Clustering adaptivity
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This section should only be executed if the macroscale loading increment where the
        # clustering adaptivity condition is triggered is to be repeated considering the new
        # clustering
        if is_crve_adaptivity and is_adapt_repeat_inc and \
                adaptivity_manager.check_inc_adaptive_steps(inc):
            # Display increment data
            if is_clust_adapt_output:
                info.displayinfo('12', crve.get_adaptive_step() + 1)
            # Build clusters equilibrium residuals dictionary
            clusters_residuals_mf = eqff.build_clusters_residuals_dict(comp_order,
                material_phases, phase_clusters, residual)
            # Get clustering adaptivity trigger condition and target clusters
            is_trigger, target_clusters, target_clusters_data = \
                adaptivity_manager.get_target_clusters(phase_clusters,
                                                       crve.get_voxels_clusters(),
                                                       clusters_state, clusters_state_old,
                                                       clusters_sct_mf, clusters_sct_mf_old,
                                                       clusters_residuals_mf,
                                                       inc, verbose=is_clust_adapt_output)
            # Perform clustering adaptivity if adaptivity condition is triggered
            if is_trigger:
                # Display clustering adaptivity
                info.displayinfo('16', 'repeat', inc)
                # Set improved initial iterative guess for the clusters incremental strain
                # global vector (matricial form) after the clustering adaptivity
                is_improved_init_guess = True
                improved_init_guess = [is_improved_init_guess, gbl_inc_strain_mf]
                # Perform clustering adaptivity
                adaptivity_manager.adaptive_refinement(crve, target_clusters,
                                                       target_clusters_data,
                                                       [clusters_state, clusters_state_old,
                                                       clusters_D_mf, clusters_De_mf],
                                                       inc, improved_init_guess,
                                                       verbose=is_clust_adapt_output)
                # Get improved initial iterative guess for the clusters incremental strain
                # global vector (matricial form)
                if is_improved_init_guess:
                    gbl_inc_strain_mf = improved_init_guess[1]
                # Update clustering dictionary
                clst_dict['voxels_clusters'] = crve.get_voxels_clusters()
                clst_dict['phase_n_clusters'] = crve.get_phase_n_clusters()
                clst_dict['phase_clusters'] = copy.deepcopy(crve.phase_clusters)
                clst_dict['clusters_f'] = copy.deepcopy(crve.clusters_f)
                # Get clusters data
                phase_n_clusters = clst_dict['phase_n_clusters']
                phase_clusters = clst_dict['phase_clusters']
                clusters_f = clst_dict['clusters_f']
                # Get total number of clusters
                n_total_clusters = crve.get_n_total_clusters()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get increment counter
                inc = mac_load_path.increm_state['inc']
                # Display increment data
                displayincdata(mac_load_path)
                # Set increment initial time
                inc_init_time = time.time()
                # Start new macroscale loading increment solution procedures
                continue
        #
        #                                              Homogenized strain and stress tensors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build homogenized strain tensor
        hom_strain = mop.gettensorfrommf(hom_strain_mf, n_dim, comp_order)
        # Build homogenized stress tensor
        hom_stress = mop.gettensorfrommf(hom_stress_mf, n_dim, comp_order)
        # Compute homogenized out-of-plane stress component in a 2D plane strain problem /
        # strain component in a 2D plane stress problem (output purpose only)
        if problem_type == 1:
            hom_stress_33 = hom.homoutofplanecomp(problem_type, material_phases,
                                                  phase_clusters, clusters_f,
                                                  clusters_state)
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
        #                                         Increment homogenized results (.hres file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize homogenized results dictionary
        hom_results = dict()
        # Build homogenized results dictionary
        hom_results['hom_strain'] = hom_strain
        hom_results['hom_stress'] = hom_stress
        if problem_type == 1:
            hom_results['hom_stress_33'] = hom_stress_33
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set post-processing procedure initial time
        procedure_init_time = time.time()
        # Write increment homogenized results (.hres)
        hres_output.write_hres_file(problem_type, inc, hom_results)
        # Increment post-processing time
        ons_post_process_time += time.time() - procedure_init_time
        #
        #                                                                 Increment VTK file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file associated to the macroscale loading increment
        if is_VTK_output and inc % vtk_inc_div == 0:
            # Set post-processing procedure initial time
            procedure_init_time = time.time()
            # Write VTK file associated to the converged increment
            vtk_output.write_VTK_file_time_step(time_step=inc,
                problem_type=problem_type, crve=crve, clusters_state=clusters_state,
                    material_phases_models=material_phases_models, vtk_vars=vtk_vars,
                        adaptivity_manager=adaptivity_manager)
            # Increment post-processing time
            ons_post_process_time += time.time() - procedure_init_time
        #
        #                                                          Converged state variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the last increment converged state variables
        clusters_state_old = copy.deepcopy(clusters_state)
        # Update the last increment converged homogenized strain and stress tensors
        # (matricial form)
        hom_strain_old_mf = hom_strain_mf
        hom_stress_old_mf = hom_stress_mf
        if problem_type == 1:
            hom_stress_33_old = hom_stress_33
        # Update last increment converged clusters strain concentration tensors
        clusters_sct_mf_old = copy.deepcopy(clusters_sct_mf)
        #
        #                                    Converged reference material elastic properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save converged reference material elastic properties from first macroscale
        # loading increment
        if inc == 1:
            mat_prop_ref_first = copy.deepcopy(mat_prop_ref)
        # Compute the reference material elastic tangent (matricial form)
        # and compliance tensor (matrix)
        # Note: This operation is not strictly necessary but is left here as a safeguard for
        #       potential future changes of the self-consistent scheme
        De_ref_mf, Se_ref_matrix = scs.refelastictanmod(problem_dict, mat_prop_ref)
        # Update converged reference material elastic properties
        mat_prop_ref_old = copy.deepcopy(mat_prop_ref)
        #
        #                                         Clustering adaptivity output (.adapt file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update clustering adaptivity output file with converged increment data
        if is_crve_adaptivity:
            # Set post-processing procedure initial time
            procedure_init_time = time.time()
            # Update clustering adaptivity output file
            adapt_output.write_adapt_file(inc, adaptivity_manager, crve, mode='increment')
            # Increment post-processing time
            ons_post_process_time += time.time() - procedure_init_time
        #
        #                        Voxels cluster state based quantities output (.voxout file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update clustering adaptivity output file with converged increment data
        if is_voxels_output:
            # Set post-processing procedure initial time
            procedure_init_time = time.time()
            # Update clustering adaptivity output file
            voxels_output.write_voxels_output_file(n_dim, comp_order, crve, clusters_state)
            # Increment post-processing time
            ons_post_process_time += time.time() - procedure_init_time
        #
        #                                                Incremental macroscale loading flow
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update converged macroscale (homogenized) state
        mac_load_path.update_hom_state(n_dim, comp_order, hom_strain, hom_stress)
        # Display converged increment data
        if problem_type == 1:
            info.displayinfo('7', 'end', problem_type, hom_strain, hom_stress,
                             time.time() - inc_init_time, time.time() - ons_init_time,
                             hom_stress_33)
        else:
            info.displayinfo('7', 'end', problem_type, hom_strain, hom_stress,
                             time.time() - inc_init_time, time.time() - ons_init_time)
        # Return if last macroscale loading increment, otherwise setup new macroscale
        # loading increment
        if is_last_inc:
            # Get total online stage time
            ons_total_time = time.time() - ons_init_time
            # Get total online stage effective time
            os_effective_time = ons_total_time - ons_post_process_time
            # Output clustering adaptivity summary
            if is_crve_adaptivity:
                info.displayinfo('15', adaptivity_manager, crve, os_effective_time)
            # Finish online stage
            return [ons_total_time, os_effective_time]
        else:
            # Setup new macroscale loading increment
            inc_mac_load_mf, n_presc_strain, presc_strain_idxs, n_presc_stress, \
                presc_stress_idxs, is_last_inc = mac_load_path.new_load_increment(
                    n_dim, comp_order)
            # Get increment counter
            inc = mac_load_path.increm_state['inc']
            # Display increment data
            displayincdata(mac_load_path)
            # Set increment initial time
            inc_init_time = time.time()
        #
        #                                                              Clustering adaptivity
        #                                                                      (deactivated)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This section should only be executed if the clustering adaptivity is to be
        # performed in between macroscale loading increments, i.e., the new clustering is
        # only considered in the following macroscale loading increments.
        if is_crve_adaptivity and not is_adapt_repeat_inc and \
                adaptivity_manager.check_inc_adaptive_steps(inc):
            # Display increment data
            if is_clust_adapt_output:
                info.displayinfo('12', crve.get_adaptive_step() + 1)
            # Build clusters equilibrium residuals dictionary
            clusters_residuals_mf = eqff.build_clusters_residuals_dict(comp_order,
                material_phases, phase_clusters, residual)
            # Get clustering adaptivity trigger condition and target clusters
            is_trigger, target_clusters, target_clusters_data = \
                adaptivity_manager.get_target_clusters(phase_clusters,
                                                       crve.get_voxels_clusters(),
                                                       clusters_state, clusters_state_old,
                                                       clusters_sct_mf, clusters_sct_mf_old,
                                                       clusters_residuals_mf,
                                                       inc, verbose=is_clust_adapt_output)
            # Perform clustering adaptivity if adaptivity condition is triggered
            if is_trigger:
                # Display clustering adaptivity
                info.displayinfo('16', 'new', inc)
                # Perform clustering adaptivity
                adaptivity_manager.adaptive_refinement(crve, target_clusters,
                                                       target_clusters_data,
                                                       [clusters_state, clusters_state_old,
                                                       clusters_D_mf, clusters_De_mf],
                                                       inc, verbose=is_clust_adapt_output)
                # Update clustering dictionary
                clst_dict['voxels_clusters'] = crve.get_voxels_clusters()
                clst_dict['phase_n_clusters'] = crve.get_phase_n_clusters()
                clst_dict['phase_clusters'] = copy.deepcopy(crve.phase_clusters)
                clst_dict['clusters_f'] = copy.deepcopy(crve.clusters_f)
                # Get clusters data
                phase_n_clusters = clst_dict['phase_n_clusters']
                phase_clusters = clst_dict['phase_clusters']
                clusters_f = clst_dict['clusters_f']
                # Get total number of clusters
                n_total_clusters = crve.get_n_total_clusters()
#
#                                                       Macroscale loading increment display
# ==========================================================================================
# Display macroscale loading increment data
def displayincdata(mac_load_path):
    # Get increment counter
    inc = mac_load_path.increm_state['inc']
    # Get loading subpath data
    sp_id, sp_inc, sp_total_lfact, sp_inc_lfact, sp_total_time, sp_inc_time, \
        subinc_level = mac_load_path.get_subpath_state()
    # Display increment data
    info.displayinfo('7', 'init', inc, subinc_level, sp_id + 1, sp_total_lfact,
                     sp_total_time, sp_inc, sp_inc_lfact, sp_inc_time)
