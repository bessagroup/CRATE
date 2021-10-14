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
import links.linksutilities as linksutil
# Links interface classes
from links.interface.links import Links
from links.interface.conversion import Reference
#
#                                                State update and consistent tangent modulus
# ==========================================================================================
# For a given increment of strain, perform the update of the material state variables and
# compute the associated consistent tangent modulus
def suct(problem_dict, clst_dict, material_properties, material_phases_models, mat_phase,
         inc_strain, state_variables_old):
    # Get problem parameters
    strain_formulation = problem_dict['strain_formulation']
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    # Set Links parameters
    links_comp_order_sym, _ = linksutil.getlinkscomporder(n_dim)
    # Get Links compiled binary (compilation with 'python' flag)
    links = Links(bin=clst_dict['links_dict']['Links_python_bin_path'])
    # Set check sent/received data flag
    is_check_data = False
    #
    #                                                             Links interfaces arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set determinant of deformation gradient (only finite strains formulation)
    detf = 1.0
    # Set state update failure flag
    sufail = False
    # Set thickness of the current Gauss point (only plane stress)
    thkgp = 0.0
    # Set the incremental strain tensor according to strain formulation
    if strain_formulation == 'infinitesimal':
        if problem_type == 1:
            eincr = np.zeros(4)
            eincr[0:3] = linksutil.gettensormflinks(inc_strain, n_dim,
                                                               links_comp_order_sym,
                                                               'strain')
        else:
            eincr = linksutil.gettensormflinks(inc_strain, n_dim,
                                                          links_comp_order_sym, 'strain')
        fincr = np.zeros((3,3))
    # Get Links material properties arrays
    iprops = material_properties[mat_phase]['iprops']
    rprops = material_properties[mat_phase]['rprops']
    # Build Links Gauss point variables arrays
    stres_py, rstava_py, lalgva_py, ralgva_py = \
        material_phases_models[mat_phase]['linksxxxxva']('set', problem_dict,
                                                         state_variables_old)
    rstav2 = copy.deepcopy(rstava_py)
    # Set required Links module variables
    nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, nadim_py, niprop_py, \
        nrprop_py, nrstav_py, nlalgv_py, nralgv_py = setrequiredmodulevars(
            problem_dict, iprops, rprops, rstava_py, lalgva_py, ralgva_py)
    # Set Links consistent tangent modulus
    dmatx_py, amatx_py = linksxmatx('set', nddim_py, nadim_py)
    #
    #                                                           Links state update interface
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data sent to the Links interface
    if is_check_data:
        checksentdata('matisu',
                      nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, niprop_py,
                      nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, sufail, thkgp,
                      eincr, fincr, iprops, lalgva_py, ralgva_py, rprops, rstava_py,
                      stres_py)
    # Set required references (arrays that are updated)
    stres = Reference(stres_py)
    rstava = Reference(rstava_py)
    lalgva = Reference(lalgva_py)
    ralgva = Reference(ralgva_py)
    # Call Links state update interface
    links.matisu_py(nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, niprop_py,
                    nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, sufail, thkgp, eincr,
                    fincr, iprops, lalgva, ralgva, rprops, rstava, stres)
    # Get updated arrays
    stres_py = stres.value
    rstava_py = rstava.value
    lalgva_py = lalgva.value
    ralgva_py = ralgva.value
    # Check data received from the Links interface
    if is_check_data:
        checkreceiveddata('matisu', stres_py, rstava_py, lalgva_py, ralgva_py)
    #
    #                                             Links consistent tangent modulus interface
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check data sent to the Links interface
    if is_check_data:
        checksentdata('matict',
                      nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, nadim_py,
                      niprop_py, nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, amatx_py,
                      dmatx_py, eincr, fincr, iprops, lalgva_py, ralgva_py, rprops,
                      rstava_py, rstav2, stres_py)
    # Set required references (arrays that are updated)
    dmatx = Reference(dmatx_py)
    amatx = Reference(amatx_py)
    # Call Links consistent tangent modulus interface
    links.matict_py(nlarge_py, ntype_py, nstre_py, nstra_py, nddim_py, nadim_py, niprop_py,
                    nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, amatx, dmatx, eincr,
                    fincr, iprops, lalgva, ralgva, rprops, rstava, rstav2, stres_py)
    # Get updated arrays
    dmatx_py = dmatx.value
    amatx_py = amatx.value
    # Check data received from the Links interface
    if is_check_data:
        checkreceiveddata('matict', dmatx_py, amatx_py)
    #
    #                                                                           State update
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get updated state variables from Links arrays
    state_variables = material_phases_models[mat_phase]['linksxxxxva'](
        'get', problem_dict, material_properties[mat_phase], stres_py, rstava_py, lalgva_py,
        ralgva_py)
    #
    #                                                             Consistent tangent modulus
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get consistent tangent modulus from associated Links array
    consistent_tangent_mf = linksxmatx('get', problem_dict, dmatx_py, amatx_py)
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return updated state variables and consistent tangent modulus
    return [state_variables, consistent_tangent_mf]
#
#                                                                   Links consistent tangent
# ==========================================================================================
# Set Links consistent tangent modulus (initialization) or get the consistent tangent
# modulus from the associated Links array
def linksxmatx(mode,*args):
    # Initialize Links consistent tangent modulus
    if mode == 'set':
        # Unpack input arguments
        nddim = args[0]
        nadim = args[1]
        # Initialize consistent tangent modulus
        dmatx = np.zeros((nddim, nddim))
        amatx = np.zeros((nadim, nadim))
        # Return
        return [dmatx, amatx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get consistent tangent modulus from Links array
    elif mode == 'get':
        # Unpack input arguments
        problem_dict = args[0]
        dmatx = args[1]
        amatx = args[2]
        # Get problem data
        problem_type = problem_dict['problem_type']
        n_dim = problem_dict['n_dim']
        comp_order_sym = problem_dict['comp_order_sym']
        # Get consistent tangent modulus (matricial form)
        if problem_type == 1:
            consistent_tangent_mf = mop.get_tensor_mf(
                linksutil.gettensorfrommflinks(dmatx[0:3, 0:3], n_dim,
                                               comp_order_sym, 'elasticity'),
                n_dim, comp_order_sym)
        else:
            consistent_tangent_mf = mop.get_tensor_mf(
                linksutil.gettensorfrommflinks(dmatx, n_dim, comp_order_sym, 'elasticity'),
                n_dim, comp_order_sym)
        # Return
        return consistent_tangent_mf
#
#                                                                     Links module variables
# ==========================================================================================
def setrequiredmodulevars(problem_dict, iprops_py, rprops_py, rstava_py, lalgva_py,
                          ralgva_py):
    # Get problem parameters
    strain_formulation = problem_dict['strain_formulation']
    problem_type = problem_dict['problem_type']
    # Set Links strain formulation flag
    if strain_formulation == 'infinitesimal':
        nlarge = 0
    else:
        nlarge = 1
    # Set Links problem type
    problem_type_converter = {'1': 2, '2': 1, '3': 3, '4': 6}
    ntype = problem_type_converter[str(problem_type)]
    # Set dimensions associated to Links
    if ntype in [1, 2, 3]:
        ndim, nstre, nstra, nddim, nadim = 2, 4, 4, 4, 5
    else:
        ndim, nstre, nstra, nddim, nadim = 3, 6, 6, 6, 9
    # Set dimensions associated to material properties arrays
    niprop = len(iprops_py)
    nrprop = len(rprops_py)
    # Set dimensions associated to Gauss point variables arrays
    nrstav = len(rstava_py)
    nlalgv = len(lalgva_py)
    nralgv = len(ralgva_py)
    # Return
    return [nlarge, ntype, ndim, nstre, nstra, nddim, nadim, niprop, nrprop, nrstav, nlalgv,
            nralgv]
#
#                                                                       Data check functions
# ==========================================================================================
# Write all the variables that are being sent to the Links interfaces to either perform the
# material state update or compute the consistent tangent (output to default stdout)
def checksentdata(interface, *args):
    if interface == 'matisu':
        # Unpack variables
        nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, niprop_py, nrprop_py, \
            nrstav_py, nlalgv_py, nralgv_py, detf, sufail, thkgp, eincr, fincr, iprops, \
            lalgva_py, ralgva_py, rprops, rstava_py, stres_py = args
        # Write variables
        print('\n' + 78*'>' + ' SEND (PYTHON)')
        print('\n' + 'linkssuct.py (state update):')
        print(       '----------------------------')
        print('\n' + '> Parameters and module variables:' + '\n')
        print('nlarge_py = ', nlarge_py)
        print('ntype_py  = ', ntype_py)
        print('ndim_py   = ', ndim_py)
        print('nstre_py  = ', nstre_py)
        print('nstra_py  = ', nstra_py)
        print('nddim_py  = ', nddim_py)
        print('niprop_py = ', niprop_py)
        print('nrprop_py = ', nrprop_py)
        print('nrstav_py = ', nrstav_py)
        print('nlalgv_py = ', nlalgv_py)
        print('nralgv_py = ', nralgv_py)
        print('\n' + '> Arguments:')
        print('\n' + 'detf      = ', detf)
        print('\n' + 'sufail    = ', sufail)
        print('\n' + 'thkgp     = ', thkgp)
        print('\n' + 'eincr     = ', eincr)
        print('\n' + 'fincr     = ', fincr[0, :])
        print(       '            ', fincr[1, :])
        print(       '            ', fincr[2, :])
        print('\n' + 'iprops    = ', iprops)
        print('\n' + 'rprops    = ', rprops)
        print('\n' + 'stres_py  = ', stres_py)
        print('\n' + 'rstava_py = ', rstava_py)
        print('\n' + 'lalgva_py = ', lalgva_py)
        print('\n' + 'ralgva_py = ', ralgva_py)
        print('\n' + 92*'>')
        print('\n')
    elif interface == 'matict':
        # Unpack variables
        nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, nadim_py, niprop_py, \
            nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, amatx_py, dmatx_py, eincr, \
            fincr, iprops, lalgva_py, ralgva_py, rprops, rstava_py, rstav2, stres_py = args
        # Write variables
        print('\n' + 78*'>' + ' SEND (PYTHON)')
        print('\n' + 'linkssuct.py (consistent tangent modulus):')
        print(       '------------------------------------------')
        print('\n' + '> Parameters and module variables:' + '\n')
        print('nlarge_py = ', nlarge_py)
        print('ntype_py  = ', ntype_py)
        print('ndim_py   = ', ndim_py)
        print('nstre_py  = ', nstre_py)
        print('nstra_py  = ', nstra_py)
        print('nddim_py  = ', nddim_py)
        print('nddim_py  = ', nadim_py)
        print('niprop_py = ', niprop_py)
        print('nrprop_py = ', nrprop_py)
        print('nrstav_py = ', nrstav_py)
        print('nlalgv_py = ', nlalgv_py)
        print('nralgv_py = ', nralgv_py)
        print('\n' + '> Arguments:')
        print('\n' + 'detf      = ', detf)
        print('\n' + 'eincr     = ', eincr)
        print('\n' + 'fincr     = ', fincr[0, :])
        print(       '            ', fincr[1, :])
        print(       '            ', fincr[2, :])
        print('\n' + 'iprops    = ', iprops)
        print('\n' + 'rprops    = ', rprops)
        print('\n' + 'stres_py  = ', stres_py)
        print('\n' + 'rstava_py = ', rstava_py)
        print('\n' + 'lalgva_py = ', lalgva_py)
        print('\n' + 'ralgva_py = ', ralgva_py)
        if dmatx_py.shape[0] == 4 and amatx_py.shape[0] == 5:
            print('\n' + 'dmatx_py = ', dmatx_py[0, :])
            print(       '           ', dmatx_py[1, :])
            print(       '           ', dmatx_py[2, :])
            print(       '           ', dmatx_py[3, :])
            print('\n' + 'amatx_py = ', amatx_py[0, :])
            print(       '           ', amatx_py[1, :])
            print(       '           ', amatx_py[2, :])
            print(       '           ', amatx_py[3, :])
            print(       '           ', amatx_py[4, :])
        elif dmatx_py.shape[0] == 6 and amatx_py.shape[0] == 9:
            print('\n' + 'dmatx_py = ', dmatx_py[0, :])
            print(       '           ', dmatx_py[1, :])
            print(       '           ', dmatx_py[2, :])
            print(       '           ', dmatx_py[3, :])
            print(       '           ', dmatx_py[4, :])
            print(       '           ', dmatx_py[5, :])
            print('\n' + 'amatx_py = ', amatx_py[0, :])
            print(       '           ', amatx_py[1, :])
            print(       '           ', amatx_py[2, :])
            print(       '           ', amatx_py[3, :])
            print(       '           ', amatx_py[4, :])
            print(       '           ', amatx_py[5, :])
            print(       '           ', amatx_py[6, :])
            print(       '           ', amatx_py[7, :])
            print(       '           ', amatx_py[8, :])
        print('\n' + 92*'>')
        print('\n')
    else:
        print('Unknown Links interface!')
# ------------------------------------------------------------------------------------------
# Write all the updated variables that are being received from the Links interfaces to
# either perform the material state update or compute the consistent tangent (output to
# default stdout)
def checkreceiveddata(interface, *args):
    if interface == 'matisu':
        # Unpack variables
        stres_py, rstava_py, lalgva_py, ralgva_py = args
        # Write variables
        print('\n' + 75*'<' + ' RECEIVE (PYTHON)')
        print('\n' + 'linkssuct.py (state update):')
        print(       '----------------------------')
        print('\n' + '> Computed/Updated variables:')
        print('\n' + 'stres_py  = ', stres_py)
        print('\n' + 'rstava_py = ', rstava_py)
        print('\n' + 'lalgva_py = ', lalgva_py)
        print('\n' + 'ralgva_py = ', ralgva_py)
        print('\n' + 92*'<')
        print('\n')
    elif interface == 'matict':
        # Unpack variables
        dmatx_py, amatx_py = args
        # Write variables
        print('\n' + 75*'<' + ' RECEIVE (PYTHON)')
        print('\n' + 'linkssuct.py (consistent tangent modulus):')
        print(       '------------------------------------------')
        print('\n' + '> Computed/Updated variables:')
        if dmatx_py.shape[0] == 4 and amatx_py.shape[0] == 5:
            print('\n' + 'dmatx_py = ', dmatx_py[0, :])
            print(       '           ', dmatx_py[1, :])
            print(       '           ', dmatx_py[2, :])
            print(       '           ', dmatx_py[3, :])
            print('\n' + 'amatx_py = ', amatx_py[0, :])
            print(       '           ', amatx_py[1, :])
            print(       '           ', amatx_py[2, :])
            print(       '           ', amatx_py[3, :])
            print(       '           ', amatx_py[4, :])
        elif dmatx_py.shape[0] == 6 and amatx_py.shape[0] == 9:
            print('\n' + 'dmatx_py = ', dmatx_py[0, :])
            print(       '           ', dmatx_py[1, :])
            print(       '           ', dmatx_py[2, :])
            print(       '           ', dmatx_py[3, :])
            print(       '           ', dmatx_py[4, :])
            print(       '           ', dmatx_py[5, :])
            print('\n' + 'amatx_py = ', amatx_py[0, :])
            print(       '           ', amatx_py[1, :])
            print(       '           ', amatx_py[2, :])
            print(       '           ', amatx_py[3, :])
            print(       '           ', amatx_py[4, :])
            print(       '           ', amatx_py[5, :])
            print(       '           ', amatx_py[6, :])
            print(       '           ', amatx_py[7, :])
            print(       '           ', amatx_py[8, :])
        print('\n' + 92*'<')
        print('\n')
    else:
        print('Unknown Links interface!')
