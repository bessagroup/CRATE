#
# Links Materials Constitutive Models Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Tensorial operations
import tensorOperations as top
# Links related procedures
import Links.LinksUtilities as LinksUtil
#
#                                                         Links material constitutive models
# ==========================================================================================
# Set material procedures for a given Links constitutive model
def LinksMaterialProcedures(model_name):
    #
    #                                                                   Linear elastic model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if model_name == 'ELASTIC':
        # Set the constitutive model required material properties
        def setRequiredProperties():
            # Set required material properties
            req_material_properties = ['density','E','v']
            # Return
            return req_material_properties
        # ----------------------------------------------------------------------------------
        # Define material constitutive model state variables and build an initialized state
        # variables dictionary
        def init(problem_dict):
            # Get problem data
            n_dim = problem_dict['n_dim']
            comp_order = problem_dict['comp_order_sym']
            problem_type = problem_dict['problem_type']
            # Define constitutive model state variables (names and initialization)
            state_variables_init = dict()
            state_variables_init['e_strain_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
            state_variables_init['strain_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
            state_variables_init['stress_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
            state_variables_init['is_su_fail'] = False
            # Set additional out-of-plane stress component in a 2D plane strain problem
            # (output purpose only)
            if problem_type == 1:
                state_variables_init['stress_33'] = 0.0
            # Return initialized state variables dictionary
            return state_variables_init
        # ----------------------------------------------------------------------------------
        # Append Links constitutive model properties specification to a given data file
        def writeMaterialProperties(file_path,mat_phase,properties):
            # Open data file to append Links constitutive model properties
            data_file = open(file_path,'a')
            # Format file structure
            write_list = [mat_phase + ' ' + 'ELASTIC' + '\n'] + \
                         [(len(mat_phase) + 1)*' ' + \
                                   str('{:16.8e}'.format(properties['density'])) + '\n'] + \
                         [(len(mat_phase) + 1)*' ' + \
                                             str('{:16.8e}'.format(properties['E'])) +
                                             str('{:16.8e}'.format(properties['v'])) + '\n']
            # Append Links constitutive model properties
            data_file.writelines(write_list)
            # Close data file
            data_file.close()
        # ----------------------------------------------------------------------------------
        # Build Links constitutive model required integer and real material properties
        # arrays (must be compatible with Links rdXXXX.f90)
        def linksXPROPS(properties):
            # Get material properties
            density = properties['density']
            E = properties['E']
            v = properties['v']
            # Compute shear and bulk modulii
            G = E/(2.0*(1.0 + v))
            K = E/(3.0*(1.0 - 2.0*v))
            # Set material type and material class
            mat_type = 1
            mat_class = 1
            # Build Links IPROPS and RPROPS arrays
            IPROPS = np.zeros(2,dtype = np.int32)
            IPROPS[0] = mat_type
            IPROPS[1] = mat_class
            RPROPS = np.zeros(3,dtype = float)
            RPROPS[0] = density
            RPROPS[1] = G
            RPROPS[2] = K
            # Return
            return [IPROPS,RPROPS]
        # ----------------------------------------------------------------------------------
        # Set Links constitutive model Gauss point variables arrays (must be compatible
        # with material_mod.f90) or get the state variables from the associated Links arrays
        def linksXXXXVA(mode,problem_dict,*args):
            # Get problem parameters
            problem_type = problem_dict['problem_type']
            n_dim = problem_dict['n_dim']
            comp_order = problem_dict['comp_order_sym']
            # Get Links parameters
            Links_comp_order,_ = LinksUtil.getLinksCompOrder(n_dim)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set Links constitutive model Gauss point variables arrays
            if mode == 'set':
                # Unpack input arguments
                state_variables = args[0]
                # Set Links strain and stress dimensions
                if problem_type == 1:
                    NSTRE = 4
                    NSTRA = 4
                else:
                    NSTRE = 6
                    NSTRA = 6
                # Get Cauchy stress
                stress_mf = state_variables['stress_mf']
                # Get real state variables
                e_strain_mf = state_variables['e_strain_mf']
                # Set logical algorithmic variables
                is_su_fail = False
                is_plast = False
                # Set Links STRES, RSTAVA, LALGVA and RALGVA arrays
                i_end = len(comp_order)
                STRES = np.zeros(NSTRE)
                STRES[0:i_end] = LinksUtil.setTensorMatricialFormLinks(
                                 top.getTensorFromMatricialForm(stress_mf,n_dim,comp_order),
                                 n_dim,Links_comp_order,'stress')
                RSTAVA = np.zeros(NSTRA)
                RSTAVA[0:i_end] = LinksUtil.setTensorMatricialFormLinks(
                                  top.getTensorFromMatricialForm(e_strain_mf,n_dim,
                                                                 comp_order),
                                  n_dim,Links_comp_order,'strain')
                LALGVA = np.zeros(2,dtype = np.int32)
                LALGVA[0] = int(is_su_fail)
                LALGVA[1] = int(is_plast)
                RALGVA = np.zeros(1,dtype = float)
                # Return
                return [STRES,RSTAVA,LALGVA,RALGVA]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get state variables from the associated Links arrays
            elif mode == 'get':
                # Unpack input arguments
                properties = args[0]
                STRES = args[1]
                RSTAVA = args[2]
                LALGVA = args[3]
                RALGVA = args[4]
                # Initialize state variables dictionary
                state_variables = init(problem_dict)
                # Get state variables
                i_end = len(comp_order)
                state_variables['e_strain_mf'] = \
                    top.setTensorMatricialForm(
                        LinksUtil.getTensorFromMatricialFormLinks(RSTAVA[0:i_end],n_dim,
                                                                  Links_comp_order,
                                                                  'strain'),
                        n_dim,comp_order)
                state_variables['strain_mf'] = \
                    top.setTensorMatricialForm(
                        LinksUtil.getTensorFromMatricialFormLinks(RSTAVA[0:i_end],n_dim,
                                                                  Links_comp_order,
                                                                  'strain'),
                        n_dim,comp_order)
                state_variables['stress_mf'] = \
                    top.setTensorMatricialForm(
                        LinksUtil.getTensorFromMatricialFormLinks(STRES[0:i_end],n_dim,
                                                                  Links_comp_order,
                                                                  'strain'),
                        n_dim,comp_order)
                state_variables['is_su_fail'] = bool(LALGVA[0])
                # Compute out-of-plane stress component in a 2D plane strain problem
                # (output purpose only)
                if problem_type == 1:
                    # Get material properties
                    E = properties['E']
                    v = properties['v']
                    # Compute Lamé parameters
                    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
                    # Compute out-of-plane stress component
                    e_strain_mf = state_variables['e_strain_mf']
                    stress_33 = lam*(e_strain_mf[comp_order.index('11')] + \
                                                        e_strain_mf[comp_order.index('22')])
                    state_variables['stress_33'] = stress_33
                # Return
                return state_variables
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [setRequiredProperties,init,writeMaterialProperties,linksXPROPS,linksXXXXVA]
