#
# Von Mises (Isotropic Hardening) Constitutive Model (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the infinitesimal strain isotropic von Mises elastoplastic
# constitutive model with isotropic strain hardening.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | March 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Read user input data file
import ioput.readinputdata as rid
# Tensorial operations
import tensorOperations as top
#
#                                                               Required material properties
#                                                                    (check input data file)
# ==========================================================================================
# Set the constitutive model required material properties
#
# Material properties meaning:
#
# E   - Young modulus
# v   - Poisson ratio
# IHL - Isotropic hardening law
#
def setRequiredProperties():
    # Set required material properties
    req_material_properties = ['E','v','IHL']
    # Return
    return req_material_properties
#
#                                                                             Initialization
# ==========================================================================================
# Define material constitutive model state variables and build an initialized state
# variables dictionary
#
# List of constitutive model state variables:
#
#   e_strain_mf  | Elastic strain tensor (matricial form)
#   acc_p_strain | Accumulated plastic strain
#   strain_mf    | Total strain tensor (matricial form)
#   stress_mf    | Cauchy stress tensor (matricial form)
#   is_plast     | Plastic step flag
#   is_su_fail   | State update failure flag
#
def init(problem_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    problem_type = problem_dict['problem_type']
    # Define constitutive model state variables (names and initialization)
    state_variables_init = dict()
    state_variables_init['e_strain_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
    state_variables_init['acc_p_strain'] = 0.0
    state_variables_init['strain_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
    state_variables_init['stress_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
    state_variables_init['is_plast'] = False
    state_variables_init['is_su_fail'] = False
    # Set additional out-of-plane strain and stress components
    if problem_type == 1:
        state_variables_init['e_strain_33'] = 0.0
        state_variables_init['stress_33'] = 0.0
    # Return initialized state variables dictionary
    return state_variables_init
#
#                                                State update and consistent tangent modulus
# ==========================================================================================
# For a given increment of strain, perform the update of the material state variables and
# compute the associated consistent tangent modulus
def suct(problem_dict,algpar_dict,material_properties,mat_phase,inc_strain,
                                                                       state_variables_old):
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Build incremental strain matricial form
    inc_strain_mf = top.setTensorMatricialForm(inc_strain,n_dim,comp_order)
    # Get algorithmic parameters
    su_max_n_iterations = algpar_dict['su_max_n_iterations']
    su_conv_tol = algpar_dict['su_conv_tol']
    # Get material properties
    E = material_properties[mat_phase]['E']
    v = material_properties[mat_phase]['v']
    hardeningLaw = material_properties[mat_phase]['hardeningLaw']
    hardening_parameters = material_properties[mat_phase]['hardening_parameters']
    # Compute shear modulus
    G = E/(2.0*(1.0 + v))
    # Compute Lamé parameters
    lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
    miu = E/(2.0*(1.0 + v))
    # Get last increment converged state variables
    e_strain_old_mf = state_variables_old['e_strain_mf']
    p_strain_old_mf = state_variables_old['strain_mf'] - e_strain_old_mf
    acc_p_strain_old = state_variables_old['acc_p_strain']
    if problem_type == 1:
        e_strain_33_old = state_variables_old['e_strain_33']
    # Initialize state update failure flag
    is_su_fail = False
    # Initialize plastic step flag
    is_plast = False
    #
    #                                                                     2D > 3D Conversion
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # When the problem type corresponds to a 2D analysis, perform the state update and
    # consistent tangent computation as in the 3D case, considering the appropriate
    # out-of-plain strain and stress components
    if problem_type == 1:
        # Set 3D problem parameters
        n_dim,comp_order_sym,_ = rid.setProblemTypeParameters(4)
        comp_order = comp_order_sym
        # Build strain tensors (matricial form) by including the appropriate out-of-plain
        # components
        inc_strain_mf = top.getStrainStress3DmfFrom2Dmf(problem_dict,inc_strain_mf,0.0)
        e_strain_old_mf = \
               top.getStrainStress3DmfFrom2Dmf(problem_dict,e_strain_old_mf,e_strain_33_old)
    #
    #                                                                           State update
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,FODevProjSym = top.setIdentityTensors(n_dim)
    FODevProjSym_mf = top.setTensorMatricialForm(FODevProjSym,n_dim,comp_order)
    # Compute elastic trial strain
    e_trial_strain_mf = e_strain_old_mf + inc_strain_mf
    # Compute elastic consistent tangent modulus according to problem type and store it in
    # matricial form
    if problem_type in [1,4]:
        # 2D problem (plane strain) / 3D problem
        e_consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    e_consistent_tangent_mf = \
                           top.setTensorMatricialForm(e_consistent_tangent,n_dim,comp_order)
    # Compute trial stress
    trial_stress_mf = np.matmul(e_consistent_tangent_mf,e_trial_strain_mf)
    # Compute deviatoric trial stress
    dev_trial_stress_mf = np.matmul(FODevProjSym_mf,trial_stress_mf)
    # Compute flow vector
    if np.allclose(dev_trial_stress_mf,np.zeros(dev_trial_stress_mf.shape),atol=1e-10):
        flow_vector_mf = np.zeros(dev_trial_stress_mf.shape)
    else:
        flow_vector_mf = \
                  np.sqrt(3.0/2.0)*(dev_trial_stress_mf/np.linalg.norm(dev_trial_stress_mf))
    # Compute von Mises equivalent trial stress
    vm_trial_stress = np.sqrt(3.0/2.0)*np.linalg.norm(dev_trial_stress_mf)
    # Compute trial accumulated plastic strain
    acc_p_trial_strain = acc_p_strain_old
    # Compute trial yield stress
    yield_stress,_ = hardeningLaw(hardening_parameters,acc_p_trial_strain)
    # Check yield function
    yield_function = vm_trial_stress - yield_stress
    # If the trial stress state lies inside the von Mises yield function, then the state
    # update is purely elastic and coincident with the elastic trial state. Otherwise, the
    # state update is elastoplastic and the return-mapping system of nonlinear equations
    # must be solved in order to update the state variables
    if yield_function <= 0:
        # Update elastic strain
        e_strain_mf = e_trial_strain_mf
        # Update stress
        stress_mf = trial_stress_mf
        # Update accumulated plastic strain
        acc_p_strain = acc_p_strain_old
    else:
        # Set plastic step flag
        is_plast = True
        # Set incremental plastic multiplier initial iterative guess
        inc_p_mult = 0
        # Initialize Newton-Raphson iteration counter
        nr_iter = 0
        # Start Newton-Raphson iterative loop
        while True:
            # Compute current yield stress and hardening modulus
            yield_stress,H = \
                            hardeningLaw(hardening_parameters,acc_p_strain_old + inc_p_mult)
            # Compute return-mapping residual (scalar)
            residual = vm_trial_stress - 3.0*G*inc_p_mult - yield_stress
            # Check Newton-Raphson iterative procedure convergence
            error = abs(residual/yield_stress)
            is_converged = error < su_conv_tol
            # Control Newton-Raphson iteration loop flow
            if is_converged:
                # Leave Newton-Raphson iterative loop (converged solution)
                break
            elif nr_iter == su_max_n_iterations:
                # Maximum number of Newton-Raphson iterations reached
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00082',location.filename,location.lineno+1,
                                                        su_max_n_iterations,mat_phase,error)
            else:
                # Increment iteration counter
                nr_iter = nr_iter + 1
            # Compute return-mapping Jacobian (scalar)
            Jacobian = -3.0*G - H
            # Solve return-mapping linearized equation
            d_iter = -residual/Jacobian
            # Update incremental plastic multiplier
            inc_p_mult = inc_p_mult + d_iter
        # Update elastic strain
        e_strain_mf = e_trial_strain_mf - inc_p_mult*flow_vector_mf
        # Update stress
        stress_mf = np.matmul(e_consistent_tangent_mf,e_strain_mf)
        # Update accumulated plastic strain
        acc_p_strain = acc_p_strain_old + inc_p_mult
    # Get the out-of-plane strain and stress components
    if problem_type == 1:
        e_strain_33 = e_strain_mf[comp_order.index('33')]
        stress_33 = stress_mf[comp_order.index('33')]
    #
    #                                                                     3D > 2D Conversion
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # When the problem type corresponds to a 2D analysis, build the 2D strain and stress
    # tensors (matricial form) once the state update has been performed
    if problem_type == 1:
        # Builds 2D strain and stress tensors (matricial form) from the associated 3D
        # counterparts
        e_trial_strain_mf = top.get2DmfFrom3Dmf(problem_dict,e_trial_strain_mf)
        e_strain_mf = top.get2DmfFrom3Dmf(problem_dict,e_strain_mf)
        stress_mf = top.get2DmfFrom3Dmf(problem_dict,stress_mf)
    #
    #                                                                Updated state variables
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize state variables dictionary
    state_variables = init(problem_dict)
    # Store updated state variables in matricial form
    state_variables['e_strain_mf'] = e_strain_mf
    state_variables['acc_p_strain'] = acc_p_strain
    state_variables['strain_mf'] = e_trial_strain_mf + p_strain_old_mf
    state_variables['stress_mf'] = stress_mf
    state_variables['is_su_fail'] = is_su_fail
    state_variables['is_plast'] = is_plast
    if problem_type == 1:
        state_variables['e_strain_33'] = e_strain_33
        state_variables['stress_33'] = stress_33
    #
    #                                                             Consistent tangent modulus
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If the state update was purely elastic, then the consistent tangent modulus is the
    # elastic consistent tangent modulus. Otherwise, compute the elastoplastic consistent
    # tangent modulus
    if is_plast:
        # Compute elastoplastic consistent tangent modulus
        factor_1 = ((inc_p_mult*6.0*G**2)/vm_trial_stress)
        factor_2 = (6.0*G**2)*((inc_p_mult/vm_trial_stress) - (1.0/(3.0*G + H)))
        unit_flow_vector = \
            np.sqrt(2.0/3.0)*top.getTensorFromMatricialForm(flow_vector_mf,n_dim,comp_order)
        consistent_tangent = e_consistent_tangent - \
              factor_1*FODevProjSym + factor_2*top.dyad22(unit_flow_vector,unit_flow_vector)
    else:
        consistent_tangent = e_consistent_tangent
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = top.setTensorMatricialForm(consistent_tangent,n_dim,comp_order)
    #
    #                                                                     3D > 2D Conversion
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # When the problem type corresponds to a 2D analysis, build the 2D consistent tangent
    # modulus (matricial form) once the 3D counterpart
    if problem_type == 1:
        consistent_tangent_mf = top.get2DmfFrom3Dmf(problem_dict,consistent_tangent_mf)
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return updated state variables and consistent tangent modulus
    return [state_variables,consistent_tangent_mf]
