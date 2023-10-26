"""Output file: Voxel material-related quantities.

This module includes the class associated with the output file where
material-related quantities defined at the voxel level are stored.

Classes
-------
VoxelsOutput:
    Output file: Voxels material-related output.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
import tensor.matrixoperations as mop
from material.materialoperations import compute_spatial_log_strain, \
                                        cauchy_from_first_piola, \
                                        MaterialQuantitiesComputer
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class VoxelsOutput:
    """Output file: Voxels material-related output.

    Attributes
    ----------
    _col_width : int
        Output file column width.
    _output_variables : list[tuple]
        Each item is a tuple associated with a material-related output variable
        that contains the variable name (index 0) and the variable number of
        dimensions (index 1). This list sets the order by which the
        material-related output variables are output.
    _output_vars_dims : int
        Total number of material-related output variables dimensions.

    Methods
    -------
    init_voxels_output_file(self, crve)
        Open output file and write file header.
    write_voxels_output_file(self, n_dim, comp_order, crve, clusters_state, \
                             clusters_def_gradient_mf)
        Write output file.
    rewind_file(self, rewind_inc)
        Rewind output file.
    """
    def __init__(self, voxout_file_path, strain_formulation, problem_type):
        """Constructor.

        Parameters
        ----------
        voxout_file_path : str
            Problem '.voxout' output file path.
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        """
        self._voxout_file_path = voxout_file_path
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # Set material-related output variables names and number of dimensions
        self._output_variables = [('vm_stress', 1), ('acc_p_strain', 1),
                                  ('acc_p_energy_dens', 1)]
        # Compute total number of output variables dimensions
        self._output_vars_dims = sum([x[1] for x in self._output_variables])
        # Set column width
        self._col_width = 16
    # -------------------------------------------------------------------------
    def init_voxels_output_file(self, crve):
        """Open output file and write file header.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        """
        # Get total number of voxels
        _, n_voxels = crve.get_n_voxels()
        # Initialize voxels output array
        voxels_array = np.zeros((self._output_vars_dims, n_voxels))
        # Initialize format structure
        write_list = []
        # Append voxels material phase labels
        write_list += \
            [''.join([('{:>' + str(self._col_width) + 'd}').format(x)
                      for x in crve.get_regular_grid().flatten('F')]) + '\n']
        # Loop over and append material-related output variables
        for i in range(len(self._output_variables)):
            write_list += \
                [''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                          for x in voxels_array[i, :]]) + '\n']
        # Open voxels material-related output file (write mode) and write
        # voxels material-related output variables initial values
        open(self._voxout_file_path, 'w').writelines(write_list)
    # -------------------------------------------------------------------------
    def write_voxels_output_file(self, n_dim, comp_order, crve,
                                 clusters_state, clusters_def_gradient_mf):
        """Write output file.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order : list[str]
            Strain/Stress components (str) order.
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            to each material cluster (key, str).
        clusters_def_gradient_mf : dict
            Deformation gradient (item, numpy.ndarray (1d)) associated with
            each material cluster (key, str), stored in matricial form.
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate factory of voxels arrays
        voxels_array_factory = VoxelsArraysFactory(self._strain_formulation,
                                                   self._problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get total number of voxels
        _, n_voxels = crve.get_n_voxels()
        # Initialize voxels output array
        voxels_array = np.zeros((self._output_vars_dims, n_voxels))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize storing index
        idx = 0
        # Loop over material-related output variables
        for i in range(len(self._output_variables)):
            # Get material-related output variable name
            outvar = self._output_variables[i][0]
            # Get material-related output variable number of dimensions
            outvar_n_dims = self._output_variables[i][1]
            # Get material-related output variable voxels array list
            array_vox_list = voxels_array_factory.build_voxels_array(
                crve, outvar, clusters_state, clusters_def_gradient_mf)
            # Loop over material-related output variable dimensions
            for var_dim in range(outvar_n_dims):
                # Store material-related output variable dimension in voxels
                # output array
                voxels_array[idx + var_dim, :] = \
                    array_vox_list[var_dim].flatten('F')
            # Update storing index
            idx += outvar_n_dims
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material-related output variables and build format
        # structure
        write_list = []
        for i in range(self._output_vars_dims):
            write_list += \
                [''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                          for x in voxels_array[i, :]]) + '\n']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open voxels material-related output file (append mode) and append
        # voxels material-related output variables of current loading increment
        open(self._voxout_file_path, 'a').writelines(write_list)
    # -------------------------------------------------------------------------
    def rewind_file(self, rewind_inc):
        """Rewind output file.

        Parameters
        ----------
        rewind_inc : int
            Increment associated with the rewind state.
        """
        # Open output file and read lines (read)
        file_lines = open(self._voxout_file_path, 'r').readlines()
        # Set output file last line
        last_line = (1 + rewind_inc)*self._output_vars_dims
        # Open output file (write mode) and write data
        open(self._voxout_file_path, 'w').writelines(
            file_lines[: last_line + 1])
# =============================================================================
class VoxelsArraysFactory:
    """Build clusters state-based voxels arrays.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    available_vars : dict
        Number of dimensions (item, int) of each available cluster state-based
        variable (key, str).

    Methods
    -------
    build_voxels_array(self, crve, csbvar, clusters_state, \
                       clusters_def_gradient_mf)
        Build clusters state-based voxel array.
    """
    # -------------------------------------------------------------------------
    def __init__(self, strain_formulation, problem_type):
        """Constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type identifier (1 - Plain strain (2D), 4- Tridimensional)
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Set available cluster state-based variables and associated number of
        # dimensions
        self._available_csbvars = {'vm_stress': 1, 'vm_strain': 1,
                                   'acc_p_strain': 1, 'acc_p_energy_dens': 1}
    # -------------------------------------------------------------------------
    def build_voxels_array(self, crve, csbvar, clusters_state,
                           clusters_def_gradient_mf):
        """Build clusters state-based voxel array.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        csbvar : str
            Cluster state based voxel-defined quantity.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            with each material cluster (key, str).
        clusters_def_gradient_mf : dict
            Deformation gradient (item, numpy.ndarray (1d)) associated with
            each material cluster (key, str), stored in matricial form.

        Returns
        -------
        array_vox_list : list[numpy.ndarray (2d or 3d)]
            List that stores arrays of a given cluster state-based
            voxel-defined quantity (item, numpy.ndarray (2d or 3d) of shape
            equal to RVE regular grid discretization).
        """
        # Check availability of cluster state-based voxel-defined quantity and
        # get number of dimensions
        if csbvar not in self._available_csbvars.keys():
            raise RuntimeError('The computation of the cluster state-based '
                               'voxel-defined quantity \'' + csbvar + '\' is '
                               'not implemented.')
        else:
            csbvar_n_dims = self._available_csbvars[csbvar]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get required variables to build cluster state-based voxel arrays
        material_phases, phase_clusters, voxels_clusters = \
            crve.get_voxels_array_variables()
        # Get number of voxels in each dimension
        n_voxels_dims, _ = crve.get_n_voxels()
        # Instantiate material state computations
        csbvar_computer = MaterialQuantitiesComputer()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize cluster state-based voxel-defined quantity arrays list
        array_vox_list = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over cluster state-based voxel-defined quantity dimensions
        for csbvar_dim in range(csbvar_n_dims):
            # Initialize voxels flat array
            array_vox_flat = np.zeros(voxels_clusters.shape).flatten('F')
            # Loop over material phases
            for mat_phase in material_phases:
                # Loop over material phase clusters
                for cluster in phase_clusters[mat_phase]:
                    # Get cluster's voxels flat indexes
                    flat_idxs = np.in1d(voxels_clusters.flatten('F'),
                                        [cluster, ])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if csbvar == 'vm_stress':
                        # Get Cauchy stress tensor (matricial form)
                        if self._strain_formulation == 'infinitesimal':
                            stress_mf = \
                                clusters_state[str(cluster)]['stress_mf']
                        else:
                            # Get deformation gradient (matricial form)
                            def_gradient_mf = \
                                clusters_def_gradient_mf[str(cluster)]
                            # Build deformation gradient
                            def_gradient = mop.get_tensor_from_mf(
                                def_gradient_mf, self._n_dim,
                                self._comp_order_nsym)
                            # Get first Piola-Kirchhoff stress tensor
                            # (matricial form)
                            first_piola_stress_mf = \
                                clusters_state[str(cluster)]['stress_mf']
                            # Build first Piola-Kirchhoff stress tensor
                            first_piola_stress = mop.get_tensor_from_mf(
                                first_piola_stress_mf, self._n_dim,
                                self._comp_order_nsym)
                            # Compute Cauchy stress tensor
                            cauchy_stress = cauchy_from_first_piola(
                                def_gradient, first_piola_stress)
                            # Get Cauchy stress tensor (matricial form)
                            stress_mf = mop.get_tensor_mf(cauchy_stress,
                                                          self._n_dim,
                                                          self._comp_order_sym)
                        # Build 3D Cauchy stress tensor (matricial form)
                        if self._problem_type == 1:
                            # Get Cauchy stress tensor out-of-plain component
                            if self._strain_formulation == 'infinitesimal':
                                stress_33 = \
                                    clusters_state[str(cluster)]['stress_33']
                            else:
                                # Get Cauchy stress tensor out-of-plain
                                # component from first Piola-Kirchhoff
                                # counterpart
                                stress_33 = (1.0/np.linalg.det(def_gradient))\
                                    * clusters_state[str(cluster)]['stress_33']
                            # Build 3D Cauchy stress tensor (matricial form)
                            stress_mf = mop.get_state_3Dmf_from_2Dmf(
                                self._problem_type, stress_mf, stress_33)
                        # Compute von Mises equivalent stress
                        value = csbvar_computer.get_vm_stress(stress_mf)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    elif csbvar == 'vm_strain':
                        # Get cluster strain tensor (matricial form)
                        if self._strain_formulation == 'infinitesimal':
                            # Get infinitesimal strain tensor (matricial form)
                            strain_mf = \
                                clusters_state[str(cluster)]['strain_mf']
                        else:
                            # Get deformation gradient (matricial form)
                            def_gradient_mf = \
                                clusters_def_gradient_mf[str(cluster)]
                            # Build deformation gradient
                            def_gradient = mop.get_tensor_from_mf(
                                def_gradient_mf, self._n_dim,
                                self._comp_order_nsym)
                            # Compute spatial logarithmic strain tensor
                            log_strain = compute_spatial_log_strain(
                                def_gradient)
                            # Get spatial logarithmic strain tensor (matricial
                            # form)
                            strain_mf = mop.get_tensor_mf(
                                log_strain, self._n_dim, self._comp_order_sym)
                        # Build 3D strain tensor (matricial form)
                        if self._problem_type == 1:
                            # Get out-of-plain strain component
                            strain_33 = 0.0
                            # Build 3D strain tensor (matricial form)
                            strain_mf = mop.get_state_3Dmf_from_2Dmf(
                                self._problem_type, strain_mf, strain_33)
                        # Compute von Mises equivalent strain
                        value = csbvar_computer.get_vm_strain(strain_mf)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    elif csbvar in ['acc_p_strain', 'acc_p_energy_dens']:
                        # Get cluster quantity directly from state variables
                        # dictionary
                        if csbvar in clusters_state[str(cluster)].keys():
                            value = clusters_state[str(cluster)][csbvar]
                        else:
                            value = 0
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store cluster's voxels data
                    array_vox_flat[flat_idxs] = value
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store cluster state-based voxel-defined quantity dimension
            array_vox_list.append(array_vox_flat.reshape(n_voxels_dims,
                                                         order='F'))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return array_vox_list
