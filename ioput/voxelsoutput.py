#
# Voxels Output Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the output file where voxels material-related quantities are stored.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | May 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Matricial operations
import tensor.matrixoperations as mop
# Material-related computations
from material.materialquantities import MaterialQuantitiesComputer
#
#                                                  Voxels material-related output file class
# ==========================================================================================
class VoxelsOutput:
    '''Voxels material-related output.

    Attributes
    ----------
    _col_width : int
        Output file column width.
    _output_variables : list
        Each item is a tuple associated to a material-related output variable that contains
        the variable name (position 0) and the variable number of dimensions (position 1).
        This list sets the order by which the material-related output variables are output.
    _output_vars_dims : int
        Total number of material-related output variables dimensions
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, voxout_file_path, problem_type):
        '''Voxels material-related output constructor.

        Parameters
        ----------
        voxout_file_path : str
            Path of voxels material-related output file.
        problem_type : int
            Problem type identifier (1 - Plain strain (2D), 4- Tridimensional)
        '''
        self._voxout_file_path = voxout_file_path
        self._problem_type = problem_type
        # Set material-related output variables names and number of dimensions
        self._output_variables = [('vm_stress', 1), ('acc_p_strain', 1),
                                  ('acc_p_energy_dens', 1)]
        # Compute total number of output variables dimensions
        self._output_vars_dims = sum([x[1] for x in self._output_variables])
        # Set column width
        self._col_width = 16
    # --------------------------------------------------------------------------------------
    def init_voxels_output_file(self, crve):
        '''Open voxels material-related output file and write increment 0 (initial) data.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        '''
        # Get total number of voxels
        _, n_voxels = crve.get_n_voxels()
        # Initialize voxels output array
        voxels_array = np.zeros((self._output_vars_dims, n_voxels))
        # Initialize format structure
        write_list = []
        # Append voxels material phase labels
        write_list += [''.join([('{:>' + str(self._col_width) + 'd}').format(x)
                                for x in crve.get_regular_grid().flatten('F')])  + '\n']
        # Loop over and append material-related output variables
        for i in range(len(self._output_variables)):
            write_list += [''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                                    for x in voxels_array[i, :]])  + '\n']
        # Open reference material output file (write mode) and write voxels material-related
        # output variables initial values
        open(self._voxout_file_path, 'w').writelines(write_list)
    # --------------------------------------------------------------------------------------
    def write_voxels_output_file(self, n_dim, comp_order, crve, clusters_state):
        '''Write voxels material-related output file.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order : list
            Strain/Stress components (str) order.
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate factory of voxels arrays
        voxels_array_factory = VoxelsArraysFactory(self._problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get total number of voxels
        _, n_voxels = crve.get_n_voxels()
        # Initialize voxels output array
        voxels_array = np.zeros((self._output_vars_dims, n_voxels))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize storing index
        idx = 0
        # Loop over material-related output variables
        for i in range(len(self._output_variables)):
            # Get material-related output variable name
            outvar = self._output_variables[i][0]
            # Get material-related output variable number of dimensions
            outvar_n_dims = self._output_variables[i][1]
            # Get material-related output variable voxels array list
            array_vox_list = \
                voxels_array_factory.build_voxels_array(crve, outvar, clusters_state)

            # Loop over material-related output variable dimensions
            for var_dim in range(outvar_n_dims):
                # Store material-related output variable dimension in voxels output array
                voxels_array[idx + var_dim, :] = array_vox_list[var_dim].flatten('F')
            # Update storing index
            idx += outvar_n_dims
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material-related output variables and build format structure
        write_list = []
        for i in range(self._output_vars_dims):
            write_list += [''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                                          for x in voxels_array[i, :]]) + '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open reference material output file (append mode) and append voxels
        # material-related output variables of current macroscale loading increment
        open(self._voxout_file_path, 'a').writelines(write_list)
#
#                                                                Voxels arrays factory class
# ==========================================================================================
class VoxelsArraysFactory:
    '''Build clusters state based voxels arrays.

    Attributes
    ----------
    available_vars : dict
        Number of dimensions (item, int) of each available cluster state based variable
        (key, str).
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, problem_type):
        '''Constructor.

        Parameters
        ----------
        problem_type : int
            Problem type identifier (1 - Plain strain (2D), 4- Tridimensional)
        '''
        self._problem_type = problem_type
        self._available_csbvars = {'vm_stress': 1, 'vm_strain': 1, 'acc_p_strain': 1,
                                   'acc_p_energy_dens': 1}
    # --------------------------------------------------------------------------------------
    def build_voxels_array(self, crve, csbvar, clusters_state):
        '''Build clusters state based voxel array.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        csbvar : str
            Cluster state based voxel-defined quantity.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).

        Returns
        -------
        array_vox_list : list
            List that stores arrays of a given cluster state based voxel-defined quantity
            (item, ndarray of shape equal to RVE regular grid discretization). A scalar
            quantity is stored in position 0, while each component of a n-dimensional
            quantity is stored in the list through a suitable order.
        '''
        # Check availability of cluster state based voxel-defined quantity and get number of
        # dimensions
        if not csbvar in self._available_csbvars.keys():
            raise RuntimeError('The computation of the cluster state based ' +
                               'voxel-defined quantity \'' + csbvar + '\' is not ' +
                               'implemented.')
        else:
            csbvar_n_dims = self._available_csbvars[csbvar]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get required variables to build cluster state based voxel arrays
        material_phases, phase_clusters, voxels_clusters = crve.get_voxels_array_variables()
        # Get number of voxels in each dimension
        n_voxels_dims, _ = crve.get_n_voxels()
        # Instantiate material state computations
        csbvar_computer = MaterialQuantitiesComputer()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize cluster state based voxel-defined quantity arrays list
        array_vox_list = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over cluster state based voxel-defined quantity dimensions
        for csbvar_dim in range(csbvar_n_dims):
            # Initialize voxels flat array
            array_vox_flat = np.zeros(voxels_clusters.shape).flatten('F')
            # Loop over material phases
            for mat_phase in material_phases:
                # Loop over material phase clusters
                for cluster in phase_clusters[mat_phase]:
                    # Get cluster's voxels flat indexes
                    flat_idxs = np.in1d(voxels_clusters.flatten('F'), [cluster, ])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if csbvar == 'vm_stress':
                        # Get cluster stress tensor (matricial form)
                        stress_mf = clusters_state[str(cluster)]['stress_mf']
                        # Build 3D stress tensor (matricial form)
                        if self._problem_type == 1:
                            # Get out-of-plain stress component
                            stress_33 = clusters_state[str(cluster)]['stress_33']
                            # Build 3D stress tensor (matricial form)
                            stress_mf = mop.getstate3Dmffrom2Dmf(self._problem_type,
                                                                 stress_mf, stress_33)
                        # Compute von Mises equivalent stress
                        value = csbvar_computer.get_vm_stress(stress_mf)
                    elif csbvar == 'vm_strain':
                        # Get cluster strain tensor (matricial form)
                        strain_mf = clusters_state[str(cluster)]['strain_mf']
                        # Build 3D strain tensor (matricial form)
                        if self._problem_type == 1:
                            # Get out-of-plain strain component
                            strain_33 = 0.0
                            # Build 3D strain tensor (matricial form)
                            strain_mf = mop.getstate3Dmffrom2Dmf(self._problem_type,
                                                                 strain_mf, strain_33)
                        # Compute von Mises equivalent strain
                        value = csbvar_computer.get_vm_strain(strain_mf)
                    elif csbvar in ['acc_p_strain', 'acc_p_energy_dens']:
                        # Get cluster quantity directly from state variables dictionary
                        if csbvar in clusters_state[str(cluster)].keys():
                            value = clusters_state[str(cluster)][csbvar]
                        else:
                            value = 0
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store cluster's voxels data
                    array_vox_flat[flat_idxs] = value
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store cluster state based voxel-defined quantity dimension
            array_vox_list.append(array_vox_flat.reshape(n_voxels_dims, order='F'))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return array_vox_list
