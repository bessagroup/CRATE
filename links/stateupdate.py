#
# Links Constitutive Modeling Interface (CRATE Program)
# ==========================================================================================
# Summary:
# Material constitutive modeling interface (state update and consistent tangent modulus) of
# the multi-scale finite element code Links (Large Strain Implicit Nonlinear Analysis of
# Solids Linking Scales), developed by the CM2S research group at the Faculty of
# Engineering, University of Porto.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Nov 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Defining abstract base classes
from abc import abstractmethod
# Shallow and deep copy operations
import copy
# Working with arrays
import numpy as np
# Matricial operations
import tensor.matrixoperations as mop
# Material operations
from material.materialoperations import first_piola_from_cauchy
# Material constitutive modeling
from material.models.interface import ConstitutiveModel
# Links related procedures
from links.configuration import get_links_comp_order, get_tensor_mf_links, get_links_dims, \
                                build_xmatx, get_consistent_tangent_from_xmatx
# Links interface classes
from links.interface.links import Links
from links.interface.conversion import Reference
#
#                                                         Links constitutive model interface
# ==========================================================================================
class LinksConstitutiveModel(ConstitutiveModel):
    '''Links constitutive model interface.

    Class Attributes
    ----------------
    Links_python_bin_path : str
        Links python binary absolute path.

    Attributes
    ----------
    _strain_type : str, {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite strain
        formulation through kinematic extension (infinitesimal constitutive formulation and
        purely finite strain kinematic extension - 'finite-kinext').
    _source : str, {'crate', }
        Material constitutive model source.
    '''
    Links_python_bin_path = None
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def write_mat_properties(self, file_path, mat_phase):
        '''Append constitutive model properties to Links input data file.

        Parameters
        ----------
        file_path : str
            Links input data file path.
        mat_phase : str
            Material phase label.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def build_xprops(self):
        '''Build Links integer and real material properties arrays.

        Returns
        -------
        iprops : 1darray
            Integer material properties.
        rprops : 1darray
            Real material properties.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def build_xxxxva(self, state_variables):
        '''Build Links constitutive model variables arrays.

        Parameters
        ----------
        state_variables : dict
            Material constitutive model state variables.

        Returns
        -------
        stres : 1darray
            Cauchy stress array.
        rstava : 1darray
            Real state variables array.
        lalgva : 1darray
            Logical algorithmic variables array.
        ralgva : 1darray
            Real algorithmic variables array.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_state_variables(self, stres, rstava, lalgva, ralgva):
        '''Get state variables from Links constitutive model variables arrays.

        Parameters
        ----------
        stres : 1darray
            Cauchy stress array.
        rstava : 1darray
            Real state variables array.
        lalgva : 1darray
            Logical algorithmic variables array.
        ralgva : 1darray
            Real algorithmic variables array.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        '''
        pass
    # --------------------------------------------------------------------------------------
    def state_update(self, inc_strain, state_variables_old, def_gradient_old):
        '''Perform material constitutive model state update.

        Parameters
        ----------
        inc_strain : 2darray
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged material constitutive model state variables.
        def_gradient_old : 2darray
            Last converged deformation gradient.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        consistent_tangent_mf : ndarray
            Material constitutive model material consistent tangent modulus in matricial
            form.
        '''
        # Set check sent/received data flag
        is_check_data = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links parameters
        links_comp_order_sym, links_comp_order_nsym = get_links_comp_order(self._n_dim)
        # Get Links compiled binary (compilation with 'python' flag)
        links = Links(bin=type(self).Links_python_bin_path)
        #
        #                                                         Links interfaces arguments
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set determinant of deformation gradient (only finite strains formulation)
        if self._strain_formulation == 'infinitesimal':
            # Assume identity deformation gradient
            detf = 1.0
        else:
            # Compute deformation gradient
            def_gradient = np.matmul(inc_strain, def_gradient_old)
            # Compute determinant of deformation gradient
            detf = np.linalg.det(def_gradient)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state update failure flag
        sufail = False
        # Set thickness of the current Gauss point (only plane stress)
        thkgp = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set the incremental strain tensor according to strain formulation
        if self._strain_formulation == 'infinitesimal':
            # Set incremental infinitesimal strain tensor
            if self._problem_type == 1:
                eincr = np.zeros(4)
                eincr[0:3] = get_tensor_mf_links(inc_strain, self._n_dim,
                                                 links_comp_order_sym, 'strain')
            else:
                eincr = get_tensor_mf_links(inc_strain, self._n_dim,
                                            links_comp_order_sym, 'strain')
            # Set dummy incremental deformation gradient
            fincr = np.zeros((3,3))
        else:
            # Set incremental deformation gradient
            fincr = get_tensor_mf_links(inc_strain, self._n_dim, links_comp_order_nsym,
                                        'strain')
            # Set dummy incremental infinitesimal strain tensor
            eincr = np.zeros((3,3))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links material properties arrays
        iprops, rprops = self.build_xprops()
        # Build Links constitutive model variables arrays
        stres_py, rstava_py, lalgva_py, ralgva_py = self.build_xxxxva(state_variables_old)
        rstav2_py = copy.deepcopy(rstava_py)
        # Set required Links module variables
        nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, nadim_py, niprop_py, \
            nrprop_py, nrstav_py, nlalgv_py, nralgv_py = \
                get_links_dims(self._strain_formulation, self._problem_type, iprops, rprops,
                               rstava_py, lalgva_py, ralgva_py)
        # Set Links initialized consistent tangent modulii
        dmatx_py, amatx_py = build_xmatx(nddim_py, nadim_py)
        #
        #                                                       Links state update interface
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data sent to the Links interface
        if is_check_data:
            type(self).output_sent_data('matisu',
                nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, niprop_py,
                    nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, sufail, thkgp, eincr,
                        fincr, iprops, lalgva_py, ralgva_py, rprops, rstava_py, stres_py)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required references (arrays that are updated)
        stres = Reference(stres_py)
        rstava = Reference(rstava_py)
        lalgva = Reference(lalgva_py)
        ralgva = Reference(ralgva_py)
        # Call Links state update interface
        links.matisu_py(nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py,
                        niprop_py, nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, sufail,
                        thkgp, eincr, fincr, iprops, lalgva, ralgva, rprops, rstava, stres)
        # Get updated arrays
        stres_py = stres.value
        rstava_py = rstava.value
        lalgva_py = lalgva.value
        ralgva_py = ralgva.value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data received from the Links interface
        if is_check_data:
            type(self).output_received_data('matisu',
                stres_py, rstava_py, lalgva_py, ralgva_py)
        #
        #                                         Links consistent tangent modulus interface
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data sent to the Links interface
        if is_check_data:
            type(self).output_sent_data('matict',
                nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, nadim_py,
                    niprop_py, nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, amatx_py,
                        dmatx_py, eincr, fincr, iprops, lalgva_py, ralgva_py, rprops,
                            rstava_py, rstav2_py, stres_py)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required references (arrays that are updated)
        dmatx = Reference(dmatx_py)
        amatx = Reference(amatx_py)
        # Call Links consistent tangent modulus interface
        links.matict_py(nlarge_py, ntype_py, nstre_py, nstra_py, nddim_py, nadim_py,
                        niprop_py, nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, amatx,
                        dmatx, eincr, fincr, iprops, lalgva, ralgva, rprops, rstava,
                        rstav2_py, stres_py)
        # Get updated arrays
        dmatx_py = dmatx.value
        amatx_py = amatx.value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check data received from the Links interface
        if is_check_data:
            type(self).output_received_data('matict', dmatx_py, amatx_py)
        #
        #                                                                       State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get state variables from Links constitutive model variables arrays
        state_variables = self.get_state_variables(stres_py, rstava_py, lalgva_py,
                                                   ralgva_py)
        # Compute first Piola-Kirchhoff stress tensor from Cauchy stress tensor and update
        # state variables
        if self._strain_formulation == 'finite':
            # Get Cauchy stress tensor (matricial form)
            cauchy_stress_mf = state_variables['stress_mf']
            # Build Cauchy stress tensor
            cauchy_stress = mop.get_tensor_from_mf(cauchy_stress_mf, self._n_dim,
                                                   self._comp_order_sym)
            # Compute first Piola-Kirchhoff stress tensor
            first_piola_stress = first_piola_from_cauchy(def_gradient, cauchy_stress)
            # Get first Piola-Kirchhoff stress tensor (matricial form)
            first_piola_stress_mf = mop.get_tensor_mf(first_piola_stress, self._n_dim,
                                                      self._comp_order_nsym)
            # Get first Piola-Kirchhoff stress tensor out-of-plane component
            if self._problem_type == 1:
                first_piola_stress_33 = \
                    np.linalg.det(def_gradient)*state_variables['stress_33']
            # Update stress tensor
            state_variables['stress_mf'] = first_piola_stress_mf
            if self._problem_type == 1:
                state_variables['stress_33'] = first_piola_stress_33
        #
        #                                                         Consistent tangent modulus
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get consistent tangent modulus from associated Links array
        consistent_tangent_mf = get_consistent_tangent_from_xmatx(self._strain_formulation,
                                                                  self._problem_type,
                                                                  dmatx_py, amatx_py)
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return updated state variables and consistent tangent modulus
        return [state_variables, consistent_tangent_mf]
    # --------------------------------------------------------------------------------------
    @staticmethod
    def output_sent_data(interface, *args):
        '''Output sent data to Links material interface.

        Parameters
        ----------
        mat_interface : str, {'matisu', 'matict'}
            Links material interface.
        '''
        if interface == 'matisu':
            # Unpack Links material state update interface variables
            nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, niprop_py, \
                nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, sufail, thkgp, eincr, \
                    fincr, iprops, lalgva_py, ralgva_py, rprops, rstava_py, stres_py = args
            # Output sent variables
            print('\n' + 78*'>' + ' SEND (PYTHON)')
            print('\n' + 'links/stateupdate.py (state update):')
            print(       '------------------------------------')
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif interface == 'matict':
            # Unpack Links material consistent tangent interface variables
            nlarge_py, ntype_py, ndim_py, nstre_py, nstra_py, nddim_py, nadim_py, \
                niprop_py, nrprop_py, nrstav_py, nlalgv_py, nralgv_py, detf, amatx_py, \
                    dmatx_py, eincr, fincr, iprops, lalgva_py, ralgva_py, rprops, \
                        rstava_py, rstav2, stres_py = args
            # Output sent variables
            print('\n' + 78*'>' + ' SEND (PYTHON)')
            print('\n' + 'links/stateupdate.py (consistent tangent modulus):')
            print(       '--------------------------------------------------')
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown Links interface.')
    # --------------------------------------------------------------------------------------
    @staticmethod
    def output_received_data(interface, *args):
        '''Output received data from Links material interface.

        Parameters
        ----------
        mat_interface : str, {'matisu', 'matict'}
            Links material interface.
        '''
        if interface == 'matisu':
            # Unpack Links material state update interface variables
            stres_py, rstava_py, lalgva_py, ralgva_py = args
            # Output received variables
            print('\n' + 75*'<' + ' RECEIVE (PYTHON)')
            print('\n' + 'links/stateupdate.py (state update):')
            print(       '------------------------------------')
            print('\n' + '> Computed/Updated variables:')
            print('\n' + 'stres_py  = ', stres_py)
            print('\n' + 'rstava_py = ', rstava_py)
            print('\n' + 'lalgva_py = ', lalgva_py)
            print('\n' + 'ralgva_py = ', ralgva_py)
            print('\n' + 92*'<')
            print('\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif interface == 'matict':
            # Unpack Links material consistent tangent interface variables
            dmatx_py, amatx_py = args
            # Output received variables
            print('\n' + 75*'<' + ' RECEIVE (PYTHON)')
            print('\n' + 'links/stateupdate.py (consistent tangent modulus):')
            print(       '--------------------------------------------------')
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown Links interface.')
