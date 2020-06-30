#
# Links State Update and Consistent Tangent Modulus Interface Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the state update and consistent tangent modulus of
# the finite element code Links constitutive models.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Matricial operations
import tensor.matrixoperations as mop
# Links related procedures
import Links.LinksUtilities as LinksUtil
# Links interface classes
from Links.interface.links import Links
from Links.interface.conversion import Reference
#
#                                                State update and consistent tangent modulus
# ==========================================================================================
# For a given increment of strain, perform the update of the material state variables and
# compute the associated consistent tangent modulus
def suct(problem_dict,clst_dict,material_properties,material_phases_models,mat_phase,
                                                            inc_strain,state_variables_old):
    # Get problem parameters
    strain_formulation = problem_dict['strain_formulation']
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    # Set Links parameters
    Links_comp_order_sym,_ = LinksUtil.getLinksCompOrder(n_dim)
    # Get Links compiled binary (compilation with 'python' flag)
    links = Links(bin=clst_dict['Links_dict']['Links_python_bin_path'])
    # Set check sent/received data flag
    is_check_data = False
    #
    #                                                             Links interfaces arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set determinant of deformation gradient (only finite strains formulation)
    DETF = 1.0
    # Set state update failure flag
    SUFAIL = False
    # Set thickness of the current Gauss point (only plane stress)
    THKGP = 0.0
    # Set the incremental strain tensor according to strain formulation
    if strain_formulation == 1:
        if problem_type == 1:
            EINCR = np.zeros(4)
            EINCR[0:3] = \
                LinksUtil.setTensorMatricialFormLinks(inc_strain,n_dim,
                                                      Links_comp_order_sym,'strain')
        else:
            EINCR = LinksUtil.setTensorMatricialFormLinks(inc_strain,n_dim,
                                                          Links_comp_order_sym,'strain')
        FINCR = np.zeros((3,3))
    # Get Links material properties arrays
    IPROPS = material_properties[mat_phase]['IPROPS']
    RPROPS = material_properties[mat_phase]['RPROPS']
    # Build Links Gauss point variables arrays
    STRES_py,RSTAVA_py,LALGVA_py,RALGVA_py = \
                                     material_phases_models[mat_phase]['linksXXXXVA']('set',
                                                           problem_dict,state_variables_old)
    RSTAV2 = copy.deepcopy(RSTAVA_py)
    # Set required Links module variables
    NLARGE_py,NTYPE_py,NDIM_py,NSTRE_py,NSTRA_py,NDDIM_py,NADIM_py,NIPROP_py,NRPROP_py, \
    NRSTAV_py,NLALGV_py,NRALGV_py = \
             setRequiredModuleVars(problem_dict,IPROPS,RPROPS,RSTAVA_py,LALGVA_py,RALGVA_py)
    # Set Links consistent tangent modulus
    DMATX_py,AMATX_py = linksXMATX('set',NDDIM_py,NADIM_py)
    #
    #                                                           Links state update interface
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data sent to the Links interface
    if is_check_data:
        checkSentData('matisu',
                      NLARGE_py,NTYPE_py,NDIM_py,NSTRE_py,NSTRA_py,NDDIM_py,NIPROP_py,
                      NRPROP_py,NRSTAV_py,NLALGV_py,NRALGV_py,DETF,SUFAIL,THKGP,EINCR,FINCR,
                      IPROPS,LALGVA_py,RALGVA_py,RPROPS,RSTAVA_py,STRES_py)
    # Set required references (arrays that are updated)
    STRES = Reference(STRES_py)
    RSTAVA = Reference(RSTAVA_py)
    LALGVA = Reference(LALGVA_py)
    RALGVA = Reference(RALGVA_py)
    # Call Links state update interface
    links.matisu_py(NLARGE_py,NTYPE_py,NDIM_py,NSTRE_py,NSTRA_py,NDDIM_py,NIPROP_py,
                    NRPROP_py,NRSTAV_py,NLALGV_py,NRALGV_py,DETF,SUFAIL,THKGP,EINCR,FINCR,
                    IPROPS,LALGVA,RALGVA,RPROPS,RSTAVA,STRES)
    # Get updated arrays
    STRES_py = STRES.value
    RSTAVA_py = RSTAVA.value
    LALGVA_py = LALGVA.value
    RALGVA_py = RALGVA.value
    # Check data received from the Links interface
    if is_check_data:
        checkReceivedData('matisu',STRES_py,RSTAVA_py,LALGVA_py,RALGVA_py)
    #
    #                                             Links consistent tangent modulus interface
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data sent to the Links interface
    if is_check_data:
        checkSentData('matict',
                      NLARGE_py,NTYPE_py,NDIM_py,NSTRE_py,NSTRA_py,NDDIM_py,NADIM_py,
                      NIPROP_py,NRPROP_py,NRSTAV_py,NLALGV_py,NRALGV_py,DETF,AMATX_py,
                      DMATX_py,EINCR,FINCR,IPROPS,LALGVA_py,RALGVA_py,RPROPS,RSTAVA_py,
                      RSTAV2,STRES_py)
    # Set required references (arrays that are updated)
    DMATX = Reference(DMATX_py)
    AMATX = Reference(AMATX_py)
    # Call Links consistent tangent modulus interface
    links.matict_py(NLARGE_py,NTYPE_py,NSTRE_py,NSTRA_py,NDDIM_py,NADIM_py,NIPROP_py,
                    NRPROP_py,NRSTAV_py,NLALGV_py,NRALGV_py,DETF,AMATX,DMATX,EINCR,FINCR,
                    IPROPS,LALGVA,RALGVA,RPROPS,RSTAVA,RSTAV2,STRES_py)
    # Get updated arrays
    DMATX_py = DMATX.value
    AMATX_py = AMATX.value
    # Check data received from the Links interface
    if is_check_data:
        checkReceivedData('matict',DMATX_py,AMATX_py)
    #
    #                                                                           State update
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get updated state variables from Links arrays
    state_variables = material_phases_models[mat_phase]['linksXXXXVA']('get',problem_dict,
                      material_properties[mat_phase],STRES_py,RSTAVA_py,LALGVA_py,RALGVA_py)
    #
    #                                                             Consistent tangent modulus
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get consistent tangent modulus from associated Links array
    consistent_tangent_mf = linksXMATX('get',problem_dict,DMATX_py,AMATX_py)
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return updated state variables and consistent tangent modulus
    return [state_variables,consistent_tangent_mf]
#
#                                                                   Links consistent tangent
# ==========================================================================================
# Set Links consistent tangent modulus (initialization) or get the consistent tangent
# modulus from the associated Links array
def linksXMATX(mode,*args):
    # Initialize Links consistent tangent modulus
    if mode == 'set':
        # Unpack input arguments
        NDDIM = args[0]
        NADIM = args[1]
        # Initialize consistent tangent modulus
        DMATX = np.zeros((NDDIM,NDDIM))
        AMATX = np.zeros((NADIM,NADIM))
        # Return
        return [DMATX,AMATX]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get consistent tangent modulus from Links array
    elif mode == 'get':
        # Unpack input arguments
        problem_dict = args[0]
        DMATX = args[1]
        AMATX = args[2]
        # Get problem data
        problem_type = problem_dict['problem_type']
        n_dim = problem_dict['n_dim']
        comp_order_sym = problem_dict['comp_order_sym']
        # Get consistent tangent modulus (matricial form)
        if problem_type == 1:
            consistent_tangent_mf = \
                mop.gettensormf(LinksUtil.getTensorFromMatricialFormLinks(
                     DMATX[0:3,0:3],n_dim,comp_order_sym,'elasticity'),n_dim,comp_order_sym)
        else:
            consistent_tangent_mf = \
                mop.gettensormf(LinksUtil.getTensorFromMatricialFormLinks(
                              DMATX,n_dim,comp_order_sym,'elasticity'),n_dim,comp_order_sym)
        # Return
        return consistent_tangent_mf
#
#                                                                     Links module variables
# ==========================================================================================
def setRequiredModuleVars(problem_dict,IPROPS_py,RPROPS_py,RSTAVA_py,LALGVA_py,RALGVA_py):
    # Get problem parameters
    strain_formulation = problem_dict['strain_formulation']
    problem_type = problem_dict['problem_type']
    # Set Links strain formulation flag
    NLARGE = 0 if strain_formulation == 1 else 1
    # Set Links problem type
    problem_type_converter = {'1':2, '2':1, '3':3, '4':6}
    NTYPE = problem_type_converter[str(problem_type)]
    # Set dimensions associated to Links
    if NTYPE in [1,2,3]:
        NDIM, NSTRE, NSTRA, NDDIM, NADIM = 2, 4, 4, 4, 5
    else:
        NDIM, NSTRE, NSTRA, NDDIM, NADIM = 3, 6, 6, 6, 9
    # Set dimensions associated to material properties arrays
    NIPROP = len(IPROPS_py)
    NRPROP = len(RPROPS_py)
    # Set dimensions associated to Gauss point variables arrays
    NRSTAV = len(RSTAVA_py)
    NLALGV = len(LALGVA_py)
    NRALGV = len(RALGVA_py)
    # Return
    return [NLARGE,NTYPE,NDIM,NSTRE,NSTRA,NDDIM,NADIM,NIPROP,NRPROP,NRSTAV,NLALGV,NRALGV]
#
#                                                                       Data check functions
# ==========================================================================================
# Write all the variables that are being sent to the Links interfaces to either perform the
# material state update or compute the consistent tangent (output to default stdout)
def checkSentData(interface,*args):
    if interface == 'matisu':
        # Unpack variables
        NLARGE_py,NTYPE_py,NDIM_py,NSTRE_py,NSTRA_py,NDDIM_py,NIPROP_py,NRPROP_py, \
            NRSTAV_py,NLALGV_py,NRALGV_py,DETF,SUFAIL,THKGP,EINCR,FINCR,IPROPS, \
            LALGVA_py,RALGVA_py,RPROPS,RSTAVA_py,STRES_py = args
        # Write variables
        print('\n' + 78*'>' + ' SEND (PYTHON)')
        print('\n' + 'LinksSUCT.py (state update):')
        print(       '----------------------------')
        print('\n' + '> Parameters and module variables:' + '\n')
        print('NLARGE_py = ', NLARGE_py)
        print('NTYPE_py  = ', NTYPE_py)
        print('NDIM_py   = ', NDIM_py)
        print('NSTRE_py  = ', NSTRE_py)
        print('NSTRA_py  = ', NSTRA_py)
        print('NDDIM_py  = ', NDDIM_py)
        print('NIPROP_py = ', NIPROP_py)
        print('NRPROP_py = ', NRPROP_py)
        print('NRSTAV_py = ', NRSTAV_py)
        print('NLALGV_py = ', NLALGV_py)
        print('NRALGV_py = ', NRALGV_py)
        print('\n' + '> Arguments:')
        print('\n' + 'DETF      = ', DETF)
        print('\n' + 'SUFAIL    = ', SUFAIL)
        print('\n' + 'THKGP     = ', THKGP)
        print('\n' + 'EINCR     = ', EINCR)
        print('\n' + 'FINCR     = ', FINCR[0,:])
        print(       '            ', FINCR[1,:])
        print(       '            ', FINCR[2,:])
        print('\n' + 'IPROPS    = ', IPROPS)
        print('\n' + 'RPROPS    = ', RPROPS)
        print('\n' + 'STRES_py  = ', STRES_py)
        print('\n' + 'RSTAVA_py = ', RSTAVA_py)
        print('\n' + 'LALGVA_py = ', LALGVA_py)
        print('\n' + 'RALGVA_py = ', RALGVA_py)
        print('\n' + 92*'>')
        print('\n')
    elif interface == 'matict':
        # Unpack variables
        NLARGE_py,NTYPE_py,NDIM_py,NSTRE_py,NSTRA_py,NDDIM_py,NADIM_py,NIPROP_py, \
            NRPROP_py,NRSTAV_py,NLALGV_py,NRALGV_py,DETF,AMATX_py,DMATX_py,EINCR,FINCR, \
            IPROPS,LALGVA_py,RALGVA_py,RPROPS,RSTAVA_py,RSTAV2,STRES_py = args
        # Write variables
        print('\n' + 78*'>' + ' SEND (PYTHON)')
        print('\n' + 'LinksSUCT.py (consistent tangent modulus):')
        print(       '------------------------------------------')
        print('\n' + '> Parameters and module variables:' + '\n')
        print('NLARGE_py = ', NLARGE_py)
        print('NTYPE_py  = ', NTYPE_py)
        print('NDIM_py   = ', NDIM_py)
        print('NSTRE_py  = ', NSTRE_py)
        print('NSTRA_py  = ', NSTRA_py)
        print('NDDIM_py  = ', NDDIM_py)
        print('NDDIM_py  = ', NADIM_py)
        print('NIPROP_py = ', NIPROP_py)
        print('NRPROP_py = ', NRPROP_py)
        print('NRSTAV_py = ', NRSTAV_py)
        print('NLALGV_py = ', NLALGV_py)
        print('NRALGV_py = ', NRALGV_py)
        print('\n' + '> Arguments:')
        print('\n' + 'DETF      = ', DETF)
        print('\n' + 'EINCR     = ', EINCR)
        print('\n' + 'FINCR     = ', FINCR[0,:])
        print(       '            ', FINCR[1,:])
        print(       '            ', FINCR[2,:])
        print('\n' + 'IPROPS    = ', IPROPS)
        print('\n' + 'RPROPS    = ', RPROPS)
        print('\n' + 'STRES_py  = ', STRES_py)
        print('\n' + 'RSTAVA_py = ', RSTAVA_py)
        print('\n' + 'LALGVA_py = ', LALGVA_py)
        print('\n' + 'RALGVA_py = ', RALGVA_py)
        if DMATX_py.shape[0] == 4 and AMATX_py.shape[0] == 5:
            print('\n' + 'DMATX_py = ', DMATX_py[0,:])
            print(       '           ', DMATX_py[1,:])
            print(       '           ', DMATX_py[2,:])
            print(       '           ', DMATX_py[3,:])
            print('\n' + 'AMATX_py = ', AMATX_py[0,:])
            print(       '           ', AMATX_py[1,:])
            print(       '           ', AMATX_py[2,:])
            print(       '           ', AMATX_py[3,:])
            print(       '           ', AMATX_py[4,:])
        elif DMATX_py.shape[0] == 6 and AMATX_py.shape[0] == 9:
            print('\n' + 'DMATX_py = ', DMATX_py[0,:])
            print(       '           ', DMATX_py[1,:])
            print(       '           ', DMATX_py[2,:])
            print(       '           ', DMATX_py[3,:])
            print(       '           ', DMATX_py[4,:])
            print(       '           ', DMATX_py[5,:])
            print('\n' + 'AMATX_py = ', AMATX_py[0,:])
            print(       '           ', AMATX_py[1,:])
            print(       '           ', AMATX_py[2,:])
            print(       '           ', AMATX_py[3,:])
            print(       '           ', AMATX_py[4,:])
            print(       '           ', AMATX_py[5,:])
            print(       '           ', AMATX_py[6,:])
            print(       '           ', AMATX_py[7,:])
            print(       '           ', AMATX_py[8,:])
        print('\n' + 92*'>')
        print('\n')
    else:
        print('Unknown Links interface!')
# ------------------------------------------------------------------------------------------
# Write all the updated variables that are being received from the Links interfaces to
# either perform the material state update or compute the consistent tangent (output to
# default stdout)
def checkReceivedData(interface,*args):
    if interface == 'matisu':
        # Unpack variables
        STRES_py,RSTAVA_py,LALGVA_py,RALGVA_py = args
        # Write variables
        print('\n' + 75*'<' + ' RECEIVE (PYTHON)')
        print('\n' + 'LinksSUCT.py (state update):')
        print(       '----------------------------')
        print('\n' + '> Computed/Updated variables:')
        print('\n' + 'STRES_py  = ', STRES_py)
        print('\n' + 'RSTAVA_py = ', RSTAVA_py)
        print('\n' + 'LALGVA_py = ', LALGVA_py)
        print('\n' + 'RALGVA_py = ', RALGVA_py)
        print('\n' + 92*'<')
        print('\n')
    elif interface == 'matict':
        # Unpack variables
        DMATX_py,AMATX_py = args
        # Write variables
        print('\n' + 75*'<' + ' RECEIVE (PYTHON)')
        print('\n' + 'LinksSUCT.py (consistent tangent modulus):')
        print(       '------------------------------------------')
        print('\n' + '> Computed/Updated variables:')
        if DMATX_py.shape[0] == 4 and AMATX_py.shape[0] == 5:
            print('\n' + 'DMATX_py = ', DMATX_py[0,:])
            print(       '           ', DMATX_py[1,:])
            print(       '           ', DMATX_py[2,:])
            print(       '           ', DMATX_py[3,:])
            print('\n' + 'AMATX_py = ', AMATX_py[0,:])
            print(       '           ', AMATX_py[1,:])
            print(       '           ', AMATX_py[2,:])
            print(       '           ', AMATX_py[3,:])
            print(       '           ', AMATX_py[4,:])
        elif DMATX_py.shape[0] == 6 and AMATX_py.shape[0] == 9:
            print('\n' + 'DMATX_py = ', DMATX_py[0,:])
            print(       '           ', DMATX_py[1,:])
            print(       '           ', DMATX_py[2,:])
            print(       '           ', DMATX_py[3,:])
            print(       '           ', DMATX_py[4,:])
            print(       '           ', DMATX_py[5,:])
            print('\n' + 'AMATX_py = ', AMATX_py[0,:])
            print(       '           ', AMATX_py[1,:])
            print(       '           ', AMATX_py[2,:])
            print(       '           ', AMATX_py[3,:])
            print(       '           ', AMATX_py[4,:])
            print(       '           ', AMATX_py[5,:])
            print(       '           ', AMATX_py[6,:])
            print(       '           ', AMATX_py[7,:])
            print(       '           ', AMATX_py[8,:])
        print('\n' + 92*'<')
        print('\n')
    else:
        print('Unknown Links interface!')
