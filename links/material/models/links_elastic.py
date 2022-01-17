#
# Links Elastic Constitutive Model Module (CRATE Program)
# ==========================================================================================
# Summary:
# Elastic constitutive model from Links (Large Strain Implicit Nonlinear Analysis of Solids
# Linking Scales), developed by the CM2S research group at the Faculty of Engineering,
# University of Porto.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Apr 2020 | Initial coding.
# Bernardo P. Ferreira | Nov 2021 | Refactoring and OOP implementation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Matricial operations
import tensor.matrixoperations as mop
# Links related procedures
from links.configuration import get_links_comp_order, get_tensor_mf_links, \
                                get_tensor_from_mf_links
# Constitutive models
from links.stateupdate import LinksConstitutiveModel
#
#                                                                 Elastic constitutive model
# ==========================================================================================
class LinksElastic(LinksConstitutiveModel):
    '''Elastic constitutive model.

    Linear elastic constitutive model (infinitesimal strains) or Hencky hyperelastic
    constitutive model (finite strains).

    Attributes
    ----------
    _name : str
        Constitutive model name.
    _strain_type : str, {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite strain
        formulation through kinematic extension (infinitesimal constitutive formulation and
        purely finite strain kinematic extension - 'finite-kinext').
    _source : str, {'crate', }
        Material constitutive model source.
    _ndim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    '''
    def __init__(self, strain_formulation, problem_type, material_properties):
        '''Constitutive model constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        material_properties : dict
            Constitutive model material properties (key, str) values (item, int/float/bool).
        '''
        self._name = 'links_elastic'
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_properties = material_properties
        # Set strain formulation
        self._strain_type = 'finite-kinext'
        # Set source
        self._source = 'links'
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_required_properties():
        '''Get the constitutive model required material properties.

        Material properties:
        density - Density
        E - Young modulus
        v - Poisson's ratio

        Returns
        -------
        req_mat_properties : list
            List of constitutive model required material properties (str).
        '''
        # Set required material properties
        req_material_properties = ['density', 'E', 'v']
        # Return
        return req_material_properties
    # --------------------------------------------------------------------------------------
    def state_init(self):
        '''Get initialized material constitutive model state variables.

        Constitutive model state variables:
            e_strain_mf | Elastic strain tensor (matricial form)
            strain_mf   | Total strain tensor (matricial form)
            stress_mf   | Cauchy stress tensor (matricial form)
            is_plast    | Plasticity flag
            is_su_fail  | State update failure flag
            e_strain_33 | Out-of-plain strain component (2D only)
            stress_33   | Out-of-plain stress component (2D only)

        Returns
        -------
        state_variables_init : dict
            Initialized constitutive model material state variables.
        '''
        # Initialize constitutive model state variables
        state_variables_init = dict()
        state_variables_init['e_strain_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                              self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                              self._comp_order_sym)
        state_variables_init['stress_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                              self._comp_order_sym)
        state_variables_init['is_plast'] = False
        state_variables_init['is_su_fail'] = False
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = 0.0
            state_variables_init['stress_33'] = 0.0
        # ----------------------------------------------------------------------------------
        # Return
        return state_variables_init
    # --------------------------------------------------------------------------------------
    def write_mat_properties(self, file_path, mat_phase):
        '''Append constitutive model properties to Links input data file.

        Parameters
        ----------
        file_path : str
            Links input data file path.
        mat_phase : str
            Material phase label.
        '''
        # Open data file to append Links constitutive model properties
        data_file = open(file_path, 'a')
        # Format file structure
        write_list = [mat_phase + ' ' + 'ELASTIC' + '\n'] + \
                     [(len(mat_phase) + 1)*' ' + \
                      str('{:<16.8e}'.format(self._material_properties['density'])) +
                      '\n'] + \
                     [(len(mat_phase) + 1)*' ' + \
                      str('{:<16.8e}'.format(self._material_properties['E'])) +
                      str('{:<16.8e}'.format(self._material_properties['v'])) + '\n']
        # Append Links constitutive model properties
        data_file.writelines(write_list)
        # Close data file
        data_file.close()
    # --------------------------------------------------------------------------------------
    def build_xprops(self):
        '''Build Links integer and real material properties arrays.

        Returns
        -------
        iprops : 1darray
            Integer material properties.
        rprops : 1darray
            Real material properties.
        '''
        # Get material properties
        density = self._material_properties['density']
        E = self._material_properties['E']
        v = self._material_properties['v']
        # Compute shear and bulk modulii
        G = E/(2.0*(1.0 + v))
        K = E/(3.0*(1.0 - 2.0*v))
        # Set material type and material class
        mat_type = 1
        mat_class = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build Links iprops array
        iprops = np.zeros(2, dtype = np.int32)
        iprops[0] = mat_type
        iprops[1] = mat_class
        # Build Links rprops array
        rprops = np.zeros(3, dtype = float)
        rprops[0] = density
        rprops[1] = G
        rprops[2] = K
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return iprops, rprops
    # --------------------------------------------------------------------------------------
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
        # Get Links parameters
        links_comp_order_sym, _ = get_links_comp_order(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links strain and stress dimensions
        if self._problem_type == 1:
            nstre = 4
            nstra = 4
        else:
            nstre = 6
            nstra = 6
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Cauchy stress
        stress_mf = state_variables['stress_mf']
        if self._problem_type == 1:
            stress_33 = state_variables['stress_33']
        # Get real state variables
        e_strain_mf = state_variables['e_strain_mf']
        if self._problem_type == 1:
            e_strain_33 = state_variables['e_strain_33']
        # Set logical algorithmic variables
        is_plast = state_variables['is_plast']
        is_su_fail = state_variables['is_su_fail']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links stres array
        stres = np.zeros(nstre)
        idx = len(self._comp_order_sym)
        stres[0:idx] = get_tensor_mf_links(
            mop.get_tensor_from_mf(stress_mf, self._n_dim, self._comp_order_sym),
                self._n_dim, links_comp_order_sym, 'stress')
        if self._problem_type == 1:
            stres[idx] = stress_33
        # Set Links rstava array
        rstava = np.zeros(nstra)
        idx = len(self._comp_order_sym)
        rstava[0:idx] = get_tensor_mf_links(
            mop.get_tensor_from_mf(e_strain_mf, self._n_dim, self._comp_order_sym),
                self._n_dim, links_comp_order_sym, 'strain')
        if self._problem_type == 1:
            rstava[idx] = e_strain_33
        # Set Links lalgva array
        lalgva = np.zeros(2, dtype=np.int32)
        lalgva[0] = int(is_plast)
        lalgva[1] = int(is_su_fail)
        # Set Links ralgva array
        ralgva = np.zeros(1, dtype=float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [stres, rstava, lalgva, ralgva]
    # --------------------------------------------------------------------------------------
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
        # Get Links parameters
        links_comp_order_sym, _ = get_links_comp_order(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize state variables dictionary
        state_variables = self.state_init()
        # Get stress from stres
        idx = len(self._comp_order_sym)
        state_variables['stress_mf'] = mop.get_tensor_mf(
            get_tensor_from_mf_links(stres[0:idx], self._n_dim, links_comp_order_sym,
                                     'stress'), self._n_dim, self._comp_order_sym)

        if self._problem_type == 1:
            state_variables['stress_33'] = stres[idx]
        # Get real state variables from rstava
        idx = len(self._comp_order_sym)
        state_variables['e_strain_mf'] = mop.get_tensor_mf(
            get_tensor_from_mf_links(rstava[0:idx], self._n_dim, links_comp_order_sym,
                                     'strain'), self._n_dim, self._comp_order_sym)
        state_variables['strain_mf'] = mop.get_tensor_mf(
            get_tensor_from_mf_links(rstava[0:idx], self._n_dim, links_comp_order_sym,
                                     'strain'), self._n_dim, self._comp_order_sym)
        if self._problem_type == 1:
            state_variables['e_strain_33'] = rstava[idx]
        # Get logical algorithmic variables from lalgva
        state_variables['is_plast'] = bool(lalgva[0])
        state_variables['is_su_fail'] = bool(lalgva[1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return state_variables
