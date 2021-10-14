#
# Links Von Mises Constitutive Model Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the infinitesimal strain isotropic elastoplastic von Mises
# constitutive model with isotropic strain hardening from finite element code Links.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Matricial operations
import tensor.matrixoperations as mop
# Links related procedures
import links.linksutilities as LinksUtil
#
#                                                              Constitutive model procedures
# ==========================================================================================
def getlinksmodelprocedures():
    # Set the constitutive model required material properties
    def getrequiredproperties():
        # Set required material properties
        req_material_properties = ['density', 'E', 'v', 'IHL']
        # Return
        return req_material_properties
    # --------------------------------------------------------------------------------------
    # Define material constitutive model state variables and build an initialized state
    # variables dictionary
    def init(problem_dict):
        # Get problem data
        n_dim = problem_dict['n_dim']
        comp_order = problem_dict['comp_order_sym']
        problem_type = problem_dict['problem_type']
        # Define constitutive model state variables (names and initialization)
        state_variables_init = dict()
        state_variables_init['e_strain_mf'] = mop.get_tensor_mf(np.zeros((n_dim, n_dim)),
                                                                n_dim, comp_order)
        state_variables_init['acc_p_strain'] = 0.0
        state_variables_init['strain_mf'] = mop.get_tensor_mf(np.zeros((n_dim, n_dim)),
                                                              n_dim, comp_order)
        state_variables_init['stress_mf'] = mop.get_tensor_mf(np.zeros((n_dim, n_dim)),
                                                              n_dim, comp_order)
        state_variables_init['is_plast'] = False
        state_variables_init['is_su_fail'] = False
        state_variables_init['inc_p_mult'] = 0.0
        # Set additional out-of-plane strain and stress components
        if problem_type == 1:
            state_variables_init['e_strain_33'] = 0.0
            state_variables_init['stress_33'] = 0.0
        # Return initialized state variables dictionary
        return state_variables_init
    # --------------------------------------------------------------------------------------
    # Append Links constitutive model properties specification to a given data file
    def writematproperties(file_path, mat_phase, properties):
        # Get hardening curve points array
        hardening_parameters = properties['hardening_parameters']
        hardening_points = hardening_parameters['hardening_points']
        n_hardening_points = hardening_parameters['n_hardening_points']
        # Open data file to append Links constitutive model properties
        data_file = open(file_path, 'a')
        # Format file structure
        write_list = [mat_phase + ' ' + 'VON_MISES' + '\n'] + \
                     [(len(mat_phase) + 1)*' ' + \
                      str('{:<16.8e}'.format(properties['density'])) + '\n'] + \
                     [(len(mat_phase) + 1)*' ' + \
                      str('{:<16.8e}'.format(properties['E'])) +
                      str('{:<16.8e}'.format(properties['v'])) + '\n'] + \
                     [(len(mat_phase) + 1)*' ' + \
                      str('{:<5d}'.format(n_hardening_points)) + '\n'] + \
                     [(len(mat_phase) + 1)*' ' + \
                      str('{:<16.8e} {:<16.8e} \n'.format(*hardening_points[i, :])) \
                      for i in range(n_hardening_points)]
        # Append Links constitutive model properties
        data_file.writelines(write_list)
        # Close data file
        data_file.close()
    # --------------------------------------------------------------------------------------
    # Build Links constitutive model required integer and real material properties
    # arrays (must be compatible with Links rdXXXX.f90)
    def linksxprops(properties):
        # Get material properties
        density = properties['density']
        E = properties['E']
        v = properties['v']
        hardening_parameters = properties['hardening_parameters']
        hardening_points = hardening_parameters['hardening_points']
        n_hardening_points = hardening_parameters['n_hardening_points']
        # Set material type and material class
        mat_type = 4
        mat_class = 1
        # Build Links iprops array
        iprops = np.zeros(3, dtype=np.int32)
        iprops[0] = mat_type
        iprops[1] = mat_class
        iprops[2] = n_hardening_points
        # Build Links rprops array
        rprops = np.zeros(3 + 2*n_hardening_points, dtype=float)
        rprops[0] = density
        rprops[1] = E
        rprops[2] = v
        j = 3
        for i in range(n_hardening_points):
            rprops[j] = hardening_points[i, 0]
            rprops[j + 1] = hardening_points[i, 1]
            j = j + 2
        # Return
        return [iprops, rprops]
    # --------------------------------------------------------------------------------------
    # Set Links constitutive model Gauss point variables arrays (must be compatible
    # with material_mod.f90) or get the state variables from the associated Links arrays
    def linksxxxxva(mode, problem_dict, *args):
        # Get problem parameters
        problem_type = problem_dict['problem_type']
        n_dim = problem_dict['n_dim']
        comp_order = problem_dict['comp_order_sym']
        # Get Links parameters
        links_comp_order, _ = LinksUtil.getlinkscomporder(n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links constitutive model Gauss point variables arrays
        if mode == 'set':
            # Unpack input arguments
            state_variables = args[0]
            # Set Links strain and stress dimensions
            if problem_type == 1:
                nstre = 4
                nstra = 4
            else:
                nstre = 6
                nstra = 6
            # Get Cauchy stress
            stress_mf = state_variables['stress_mf']
            if problem_type == 1:
                stress_33 = state_variables['stress_33']
            # Get real state variables
            e_strain_mf = state_variables['e_strain_mf']
            if problem_type == 1:
                e_strain_33 = state_variables['e_strain_33']
            acc_p_strain = state_variables['acc_p_strain']
            # Get real algorithmic variables
            inc_p_mult = state_variables['inc_p_mult']
            # Set logical algorithmic variables
            is_plast = state_variables['is_plast']
            is_su_fail = state_variables['is_su_fail']
            # Set Links stres array
            stres = np.zeros(nstre)
            idx = len(comp_order)
            stres[0:idx] = LinksUtil.gettensormflinks(
                mop.get_tensor_from_mf(stress_mf, n_dim, comp_order), n_dim,
                    links_comp_order, 'stress')
            if problem_type == 1:
                stres[idx] = stress_33
            # Set Links rstava array
            rstava = np.zeros(nstra + 1)
            idx = len(comp_order)
            rstava[0:idx] = LinksUtil.gettensormflinks(mop.get_tensor_from_mf(
                e_strain_mf, n_dim, comp_order), n_dim, links_comp_order, 'strain')
            if problem_type == 1:
                rstava[idx] = e_strain_33
            rstava[-1] = acc_p_strain
            # Set Links lalgva array
            lalgva = np.zeros(2, dtype=np.int32)
            lalgva[0] = int(is_plast)
            lalgva[1] = int(is_su_fail)
            # Set Links ralgva array
            ralgva = np.zeros(1, dtype=float)
            ralgva[0] = inc_p_mult
            # Return
            return [stres, rstava, lalgva, ralgva]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get state variables from the associated Links arrays
        elif mode == 'get':
            # Unpack input arguments
            properties = args[0]
            stres = args[1]
            rstava = args[2]
            lalgva = args[3]
            ralgva = args[4]
            # Initialize state variables dictionary
            state_variables = init(problem_dict)
            # Get stress from stres
            idx = len(comp_order)
            state_variables['stress_mf'] = mop.get_tensor_mf(
                LinksUtil.gettensorfrommflinks(stres[0:idx], n_dim,
                links_comp_order, 'stress'), n_dim, comp_order)
            if problem_type == 1:
                state_variables['stress_33'] = stres[idx]
            # Get real state variables from rstava
            idx = len(comp_order)
            state_variables['e_strain_mf'] = mop.get_tensor_mf(
                LinksUtil.gettensorfrommflinks(rstava[0:idx], n_dim, links_comp_order,
                                               'strain'), n_dim, comp_order)
            state_variables['strain_mf'] = np.zeros_like(rstava[0:idx])
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00084', location.filename, location.lineno + 1)
            if problem_type == 1:
                state_variables['e_strain_33'] = rstava[idx]
            state_variables['acc_p_strain'] = rstava[-1]
            # Get logical algorithmic variables from lalgva
            state_variables['is_plast'] = bool(lalgva[0])
            state_variables['is_su_fail'] = bool(lalgva[1])
            # Get logical algorithmic variables from ralgva
            state_variables['inc_p_mult'] = ralgva[0]
            # Return
            return state_variables
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [getrequiredproperties, init, writematproperties, linksxprops, linksxxxxva]
