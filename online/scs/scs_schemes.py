#
# Self-Consistent Schemes Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the reference (fictitious) material and associated self-consistent
# scheme.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
# Linear elastic constitutive model
import material.models.linear_elastic
#
#                                                 Reference material elastic tangent modulus
# ==========================================================================================
# Compute the reference material elastic tangent (matricial form) and compliance tensor
# (matrix)
def refelastictanmod(problem_dict, material_properties_ref):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Compute reference material elastic tangent (matricial form)
    De_ref_mf = material.models.linear_elastic.ct(problem_dict, material_properties_ref)
    # Get reference material Young modulus and Poisson ratio
    E_ref = material_properties_ref['E']
    v_ref = material_properties_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute reference material compliance tensor
    _, foid, _, fosym, fodiagtrace, _, _ = top.getidoperators(n_dim)
    Se_ref = -(lam_ref/(2*miu_ref*(3*lam_ref + 2*miu_ref)))*fodiagtrace + \
        (1.0/(2.0*miu_ref))*fosym
    # Compute reference material compliance tensor (matricial form)
    Se_ref_mf = mop.gettensormf(Se_ref, n_dim, comp_order)
    # Store reference material compliance tensor in a matrix similar to matricial form
    # but without any associated coefficients
    Se_ref_matrix = np.zeros(Se_ref_mf.shape)
    for j in range(len(comp_order)):
        for i in range(len(comp_order)):
            Se_ref_matrix[i, j] = (1.0/mop.kelvinfactor(i, comp_order))*\
                (1.0/mop.kelvinfactor(j, comp_order))*Se_ref_mf[i, j]
    # Return
    return [De_ref_mf, Se_ref_matrix]
#
#                                                                     Self-consistent scheme
# ==========================================================================================
# Update reference material elastic properties through a given self-consistent scheme
def scsupdate(self_consistent_scheme, problem_dict, inc_strain_mf, inc_stress_mf,
              material_properties_ref, *args):
    # Get problem data
    problem_type = problem_dict['problem_type']
    # Perform self-consistent scheme to update the reference material elastic properties
    # 1. Regression-based scheme
    # 2. Projection-based scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if self_consistent_scheme == 1:
        # Select regression-based scheme option (only affects the 2D plane strain case):
        #
        # Option 1 - In a 2D plane strain problem consider the 2D parameters, i.e. a 2x2
        #            second-order identity tensor and the inplane strain/stress components;
        #
        # Option 2 - Always set the self-consistent scheme system of linear equations in a
        #            full 3D setting, i.e. a 3x3 second-order identity tensor and the 3D
        #            strain/stress tensors
        #
        scs_option = 1
        # OPTION 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if scs_option == 1:
            # Get problem data
            problem_type = problem_dict['problem_type']
            n_dim = problem_dict['n_dim']
            comp_order = problem_dict['comp_order_sym']
        # OPTION 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        elif scs_option == 2:
            # When the problem type corresponds to a 2D analysis, build the 3D strain and
            # stress tensors by considering the appropriate out-of-plain strain and stress
            # components
            if problem_type == 1:
                # Get out-of-plain stress component
                inc_stress_33 = args[0]
                # Build the incremental strain/stress tensors (matricial form) by including
                # the appropriate out-of-plain components
                inc_strain_mf = top.getstate3Dmffrom2Dmf(problem_dict, inc_strain_mf, 0.0)
                inc_stress_mf = top.getstate3Dmffrom2Dmf(problem_dict, inc_stress_mf,
                                                         inc_stress_33)
            # Solve the regression-based self-consistent scheme always considering the
            # 3D strain and stress tensors
            problem_type = 4
            n_dim, comp_order_sym, _ = mop.getproblemtypeparam(problem_type)
            comp_order = comp_order_sym
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Set second-order identity tensor
        soid, _, _, _, _, _, _ = top.getidoperators(n_dim)
        # Initialize self-consistent scheme system of linear equations coefficient matrix
        # and right-hand side
        scs_matrix = np.zeros((2,2))
        scs_rhs = np.zeros(2)
        # Get incremental strain and stress tensors
        inc_strain = mop.gettensorfrommf(inc_strain_mf, n_dim, comp_order)
        inc_stress = mop.gettensorfrommf(inc_stress_mf, n_dim, comp_order)
        # Compute self-consistent scheme system of linear equations right-hand side
        scs_rhs[0] = np.trace(inc_stress)
        scs_rhs[1] = top.ddot22_1(inc_stress, inc_strain)
        # Compute self-consistent scheme system of linear equations coefficient matrix
        scs_matrix[0, 0] = np.trace(inc_strain)*np.trace(soid)
        scs_matrix[0, 1] = 2.0*np.trace(inc_strain)
        scs_matrix[1, 0] = np.trace(inc_strain)**2
        scs_matrix[1, 1] = 2.0*top.ddot22_1(inc_strain, inc_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Limitation 1: Under isochoric loading conditions the first equation of the
        # self-consistent scheme system of linear equations vanishes (derivative with
        # respect to lambda). In this case, adopt the previous converged lambda and compute
        # miu from the second equation of the the self-consistent scheme system of linear
        # equations
        if (abs(np.trace(inc_strain))/np.linalg.norm(inc_strain)) < 1e-10 or \
            np.linalg.solve(scs_matrix, scs_rhs)[0] < 0:
            # Get previous converged reference material elastic properties
            E_ref_old = material_properties_ref['E']
            v_ref_old = material_properties_ref['v']
            # Compute previous converged lambda
            lam_ref = (E_ref_old*v_ref_old)/((1.0 + v_ref_old)*(1.0 - 2.0*v_ref_old))
            # Compute miu
            miu_ref = scs_rhs[1]/scs_matrix[1,1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Limitation 2: Under hydrostatic loading conditions both equations of the
        # self-consistent scheme system of linear equations become linearly dependent. In
        # this case, assume that the ratio between lambda and miu is the same as in the
        # previous converged values and solve the first equation of self-consistent scheme
        # system of linear equations
        elif np.all([abs(inc_strain[0, 0] - inc_strain[i, i])/np.linalg.norm(inc_strain)
                < 1e-10 for i in range(n_dim)]) and \
                np.allclose(inc_strain, np.diag(np.diag(inc_strain)), atol=1e-10):
            # Get previous converged reference material elastic properties
            E_ref_old = material_properties_ref['E']
            v_ref_old = material_properties_ref['v']
            # Compute previous converged reference material Lamé parameters
            lam_ref_old = (E_ref_old*v_ref_old)/((1.0 + v_ref_old)*(1.0 - 2.0*v_ref_old))
            miu_ref_old = E_ref_old/(2.0*(1.0 + v_ref_old))
            # Compute reference material Lamé parameters
            lam_ref = (scs_rhs[0]/scs_matrix[0,0])*(lam_ref_old/(lam_ref_old + miu_ref_old))
            miu_ref = (scs_rhs[0]/scs_matrix[0,0])*(miu_ref_old/(lam_ref_old + miu_ref_old))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve self-consistent scheme system of linear equations
        else:
            scs_solution = np.linalg.solve(scs_matrix,scs_rhs)
            # Get reference material Lamé parameters
            lam_ref = scs_solution[0]
            miu_ref = scs_solution[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute reference material Young modulus and Poisson ratio
        E_ref = (miu_ref*(3.0*lam_ref + 2.0*miu_ref))/(lam_ref + miu_ref)
        v_ref = lam_ref/(2.0*(lam_ref + miu_ref))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif self_consistent_scheme == 2:
        # Compute incremental deviatoric strain
        if problem_type == 1:
            # I strongly believe that the 2D implementation of the projection-based
            # self-consistent scheme requires the 3D cluster interaction tensors in order
            # to compute the shear modulus. Since this yields a prohibitive additional
            # computational cost, the projection-based self-consistent scheme is only
            # implemented for the 3D case
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00085',location.filename,location.lineno+1)
        else:
            # Get effective tangent modulus
            eff_tangent_mf = args[0]
            # Get problem parameters
            problem_type = problem_dict['problem_type']
            n_dim = problem_dict['n_dim']
            comp_order = problem_dict['comp_order_sym']
            # Set fourth-order deviatoric projection tensor (matricial form)
            _, _, _, _, _, _, fodevprojsym = top.getidoperators(n_dim)
            FODevProjSym_mf = mop.gettensormf(fodevprojsym, n_dim, comp_order)
            # Compute incremental deviatoric strain
            inc_dev_strain_mf = np.matmul(FODevProjSym_mf, inc_strain_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute reference material bulk modulus
            K_ref = 0.0
            for j in range(len(comp_order)):
                comp_j = comp_order[j]
                if comp_j[0] == comp_j[1]:
                    for i in range(len(comp_order)):
                        comp_i = comp_order[i]
                        if comp_i[0] == comp_i[1]:
                            K_ref = K_ref + eff_tangent_mf[i, j]/(n_dim**2)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute reference material shear modulus
            if abs(np.linalg.norm(inc_dev_strain_mf)) < 1e-10:
                # Limitation: Under pure volumetric loading conditions the incremental
                # deviatoric strain is null. In this case, assume that the ratio between
                # the material bulk modulus and shear modulus is the same as in the previous
                # converged values
                E_ref_old = material_properties_ref['E']
                v_ref_old = material_properties_ref['v']
                K_ref_old = E_ref_old/(3.0*(1.0 - 2.0*v_ref_old))
                miu_ref_old = E_ref_old/(2.0*(1.0 + v_ref_old))
                miu_ref = (miu_ref_old/K_ref_old)*K_ref
            else:
                miu_ref = np.dot(
                    np.matmul(eff_tangent_mf, inc_dev_strain_mf), inc_dev_strain_mf)/ \
                    (2.0*(np.linalg.norm(inc_dev_strain_mf)**2))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute Lamé parameter
            lam_ref = K_ref - (2.0/3.0)*miu_ref
            # Compute reference material Young modulus and Poisson ratio
            E_ref = (miu_ref*(3.0*lam_ref + 2.0*miu_ref))/(lam_ref + miu_ref)
            v_ref = lam_ref/(2.0*(lam_ref + miu_ref))
    # Return
    return [E_ref, v_ref]
#
#                                              Self-consistent scheme convergence evaluation
# ==========================================================================================
# Check self-consistent scheme iterative procedure convergence
def checkscsconvergence(E_ref, v_ref, material_properties_ref, scs_conv_tol):
    # Compute iterative variation of the reference material Young modulus and Poisson ratio
    d_E_ref = E_ref - material_properties_ref['E']
    d_v_ref = v_ref - material_properties_ref['v']
    # Compute normalized interative change of the reference material Young modulus and
    # Poisson ratio
    norm_d_E_ref = abs(d_E_ref/E_ref)
    norm_d_v_ref = abs(d_v_ref/v_ref)
    # The self-consistent scheme convergence flag is True if the normalized iterative
    # change of the reference material elastic properties converged according to the defined
    # convergence tolerance
    is_scs_converged = (norm_d_E_ref < scs_conv_tol) and (norm_d_v_ref < scs_conv_tol)
    # ------------------------------------------------------------------------------
    # Validation:
    if False:
        section = 'Self-consistent scheme convergence evaluation'
        print('\n' + '>> ' + section + ' ' + (92-len(section)-4)*'-')
        print('\n' + 'd_E_ref = ' + '{:11.4e}'.format(d_E_ref))
        print('\n' + 'd_v_ref = ' + '{:11.4e}'.format(d_v_ref))
        print('\n' + 'norm_d_E_ref = ' + '{:11.4e}'.format(norm_d_E_ref))
        print('\n' + 'norm_d_v_ref = ' + '{:11.4e}'.format(norm_d_v_ref))
    # ------------------------------------------------------------------------------
    # Return
    return [is_scs_converged, norm_d_E_ref, norm_d_v_ref]
