"""VTK (XML format) output files.

This module includes a complete toolkit to generate VTK files (XML format)
containing data defined at the microstructure level, assumed to be spatially
discretized in a regular grid of voxels. Such data may include information
about the material phases, clusters and state variables local fields, allowing
the spatial visualization through suitable VTK-reading software
(e.g., Paraview).

Classes
-------
VTKOutput
    VTK output.
VTKCollection
    VTK collection.
XMLGenerator
    VTK XML files generation methods.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import copy
# Third-party
import numpy as np
# Local
import tensor.matrixoperations as mop
from ioput.miscoutputfiles.voxelsoutput import VoxelsArraysFactory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class VTKOutput:
    """VTK output.

    Attributes
    ----------
    _vtk_collection : VTKCollection
        VTK collection output.

    Methods
    -------
    write_vtk_file_clustering(self, crve)
        Write VTK file associated with the CRVE clustering.
    write_vtk_file_time_step(self, time_step, strain_formulation, \
                             problem_type, crve, material_state, \
                             vtk_vars='all', adaptivity_manager=None)
        Write VTK file associated with time step (increment).
    rewind_files(self, rewind_inc)
        Rewind VTK output files.
    _set_image_data_parameters(rve_dims, n_voxels_dims)
        Set ImageData dataset parameters.
    _set_cell_data_name(self, comp_order_sym, comp_order_nsym, var_name, \
                        var_type, idx=None, model_name=None)
        Set variable cell data array name.
    _set_state_var_descriptors(self, n_dim, comp_order_sym, comp_order_nsym, \
                               var_name, source, model_state_variables=None, \
                               cluster=None, clusters_state=None)
        Set state variable descriptors required to output cell data array.
    _state_var_cell_data_array(self, n_dim, comp_order_sym, comp_order_nsym, \
                               material_phases, material_phases_models, \
                               phase_clusters, voxels_clusters, \
                               clusters_state, var_name, var_type, \
                               comp_idx=None, model_name=None)
        Build state variable cell data array.
    _reset_labels_from_zero(array_1d)
        Reset 1d array of integers starting from zero.
    """
    def __init__(self, type, version, byte_order, format, precision,
                 header_type, base_name, vtk_dir, pvd_dir=None):
        """Constructor.

        Parameters
        ----------
        type : str
            VTK file type.
        version : str
            VTK version.
        byte_order : str
            VTK byte order.
        format : str
            VTK file format.
        precision : {'SinglePrecision', 'DoublePrecision'}
            VTK file data precision.
        header_type : str
            VTK header type.
        base_name : str
            Base name of VTK files.
        vtk_dir : str
            Directory where VTK files are stored.
        pvd_dir : str, default=None
            Directory where PVD file is stored.
        """
        self._type = type
        self._version = version
        self._byte_order = byte_order
        self._format = format
        self._precision = precision
        self._header_type = header_type
        self._base_name = base_name
        self._vtk_dir = vtk_dir
        if pvd_dir is None:
            self._pvd_dir = self._vtk_dir
        else:
            self._pvd_dir = pvd_dir
        self._vtk_collection = None
    # -------------------------------------------------------------------------
    def write_vtk_file_clustering(self, crve):
        """Write VTK file associated with the CRVE clustering.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        """
        # Set clustering VTK file path
        vtk_file_path = self._vtk_dir + self._base_name + '_clusters.vti'
        # Open clustering VTK file (append mode)
        if os.path.isfile(vtk_file_path):
            os.remove(vtk_file_path)
        vtk_file = open(vtk_file_path, 'a')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate VTK XML generator
        xml = XMLGenerator(self._type, self._version, self._byte_order,
                           self._format, self._precision, self._header_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE dimensions
        rve_dims = crve.get_rve_dims()
        # Get regular grid of voxels
        regular_grid = crve.get_regular_grid()
        # Get number of voxels in each direction
        n_voxels_dims = [regular_grid.shape[i]
                         for i in range(len(regular_grid.shape))]
        # Get cluster associated with each pixel/voxel
        voxels_clusters = crve.get_voxels_clusters()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file header
        xml.write_file_header(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK dataset element
        dataset_parameters, piece_parameters = \
            type(self)._set_image_data_parameters(rve_dims, n_voxels_dims)
        xml.write_open_dataset_elem(vtk_file, dataset_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK dataset element piece
        xml.write_open_dataset_piece(vtk_file, piece_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK dataset element piece cell data
        xml.write_open_cell_data(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Material phases
        data_list = list(regular_grid.flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': 'Material phase', 'format': self._format,
                           'RangeMin': min_val, 'RangeMax': max_val}
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Clusters
        voxels_clusters_flat = \
            self._reset_labels_from_zero(voxels_clusters.flatten('F'))
        data_list = list(voxels_clusters_flat)
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': 'Cluster', 'format': self._format,
                           'RangeMin': min_val, 'RangeMax': max_val}
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element cell data
        xml.write_close_cell_data(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element piece
        xml.write_close_dataset_piece(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element
        xml.write_close_dataset_elem(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file footer
        xml.write_file_footer(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close clustering VTK file
        vtk_file.close()
    # -------------------------------------------------------------------------
    def write_vtk_file_time_step(self, time_step, strain_formulation,
                                 problem_type, crve, material_state,
                                 vtk_vars='all', adaptivity_manager=None):
        """Write VTK file associated with time step (increment).

        Parameters
        ----------
        time_step : int
            Time step.
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state.
        vtk_vars : {'all', 'common'}, default='all'
            If 'common', only state variables common to all material phases
            constitutive models are output. Otherwise, all state variables are
            output.
        adaptivity_manager : AdaptivityManager
            CRVE clustering adaptivity manager.
        """
        # Set VTK file path
        vtk_file_path = self._vtk_dir + self._base_name + '_' \
            + str(time_step) + '.vti'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK collection file
        if time_step == 0:
            # Set VTK collection file path
            pvd_file_path = self._pvd_dir + self._base_name + '.pvd'
            self._vtk_collection = VTKCollection(pvd_file_path=pvd_file_path)
            self._vtk_collection.init_vtk_collection_file()
        # Add VTK file path to collection
        self._vtk_collection.write_vtk_collection_file(
            time_step=time_step,
            time_step_file_path='VTK/' + vtk_file_path.split('VTK/', 1)[1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open clustering VTK file (append mode)
        if os.path.isfile(vtk_file_path):
            os.remove(vtk_file_path)
        vtk_file = open(vtk_file_path, 'a')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate VTK XML generator
        xml = XMLGenerator(self._type, self._version, self._byte_order,
                           self._format, self._precision, self._header_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE dimensions
        rve_dims = crve.get_rve_dims()
        # Set problem number of dimensions
        n_dim = len(rve_dims)
        # Get strain/stress components order
        _, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get regular grid of voxels
        regular_grid = crve.get_regular_grid()
        # Get number of voxels in each direction
        n_voxels_dims = [regular_grid.shape[i]
                         for i in range(len(regular_grid.shape))]
        # Get material phases
        material_phases = crve.get_material_phases()
        # Get clusters associated with each material phase
        phase_clusters = crve.get_phase_clusters()
        # Get cluster associated with each pixel/voxel
        voxels_clusters = crve.get_voxels_clusters()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material constitutive model associated with each material phase
        material_phases_models = material_state.get_material_phases_models()
        # Get material state variables associated with each material cluster
        clusters_state = material_state.get_clusters_state()
        # Get deformation gradient associated with each material cluster
        clusters_def_gradient_mf = \
            material_state.get_clusters_def_gradient_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate factory of voxels arrays
        voxels_array_factory = VoxelsArraysFactory(strain_formulation,
                                                   problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file header
        xml.write_file_header(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK dataset element
        dataset_parameters, piece_parameters = \
            type(self)._set_image_data_parameters(rve_dims, n_voxels_dims)
        xml.write_open_dataset_elem(vtk_file, dataset_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK dataset element piece
        xml.write_open_dataset_piece(vtk_file, piece_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK dataset element piece cell data
        xml.write_open_cell_data(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Material phases
        data_list = list(regular_grid.flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': 'Material phase', 'format': self._format,
                           'RangeMin': min_val, 'RangeMax': max_val}
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Clusters
        voxels_clusters_flat = \
            self._reset_labels_from_zero(voxels_clusters.flatten('F'))
        data_list = list(voxels_clusters_flat)
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': 'Cluster', 'format': self._format,
                           'RangeMin': min_val, 'RangeMax': max_val}
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state variables common to all material constitutive models
        if problem_type == 1:
            common_var_list = ['stress_mf', 'stress_33']
        else:
            common_var_list = ['stress_mf']
        # Loop over common state variables
        for var_name in common_var_list:
            # Initialize common state variable flag
            is_common_var = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phases
            for mat_phase in material_phases:
                # Get material phase constitutive model
                constitutive_model = material_phases_models[str(mat_phase)]
                # Get material constitutive model state variables
                model_state_variables = constitutive_model.state_init()
                # Check state variable
                if var_name not in model_state_variables.keys():
                    # Set common state variable flag
                    is_common_var = False
                    # Remove state variable from common list
                    common_var_list.remove(var_name)
                    # Skip state variable output
                    break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Output common state variable
            if is_common_var:
                # Get state variable descriptors
                var, var_type, var_n_comp = self._set_state_var_descriptors(
                    n_dim, comp_order_sym, comp_order_nsym, var_name,
                    source='model_state_variables',
                    model_state_variables=model_state_variables)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over state variable components
                for comp_idx in range(var_n_comp):
                    # Build state variable cell data array
                    rg_array = self._state_var_cell_data_array(
                        n_dim, comp_order_sym, comp_order_nsym,
                        material_phases, material_phases_models,
                        phase_clusters, voxels_clusters, clusters_state,
                        var_name, var_type, comp_idx=comp_idx)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set output variable data name
                    data_name = self._set_cell_data_name(
                        comp_order_sym, comp_order_nsym, var_name, var_type,
                        idx=comp_idx)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Write VTK cell data array - State variable
                    data_list = list(rg_array.flatten('F'))
                    min_val = min(data_list)
                    max_val = max(data_list)
                    data_parameters = {'Name': data_name,
                                       'format': self._format,
                                       'RangeMin': min_val,
                                       'RangeMax': max_val}
                    xml.write_cell_data_array(vtk_file, data_list,
                                              data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if vtk_vars == 'all':
            # Initialize list of constitutive models whose state variables have
            # already been output
            output_model_names = []
            # Loop over material phases
            for mat_phase in material_phases:
                # Get material phase constitutive model
                constitutive_model = material_phases_models[str(mat_phase)]
                # Get material constitutive model name
                model_name = constitutive_model.get_name()
                # Skip to next material phase if constitutive model state
                # variables have already been output. Otherwise, add material
                # constitutive model name to output list
                if model_name in output_model_names:
                    continue
                else:
                    output_model_names.append(model_name)
                # Get material constitutive model state variables
                model_state_variables = constitutive_model.state_init()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over constitutive model state variables
                for var_name in list(set(model_state_variables.keys())
                                     - set(common_var_list)):
                    # Get state variable descriptors
                    var, var_type, var_n_comp = \
                        self._set_state_var_descriptors(
                            n_dim, comp_order_sym, comp_order_nsym, var_name,
                            source='model_state_variables',
                            model_state_variables=model_state_variables)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over state variable components
                    for comp_idx in range(var_n_comp):
                        # Build state variable cell data array
                        rg_array = self._state_var_cell_data_array(
                            n_dim, comp_order_sym, comp_order_nsym,
                            material_phases, material_phases_models,
                            phase_clusters, voxels_clusters, clusters_state,
                            var_name, var_type, comp_idx=comp_idx,
                            model_name=model_name)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Set output variable data name
                        data_name = self._set_cell_data_name(
                            comp_order_sym, comp_order_nsym, var_name,
                            var_type, idx=comp_idx, model_name=model_name)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Write VTK cell data array - State variable
                        data_list = list(rg_array.flatten('F'))
                        min_val = min(data_list)
                        max_val = max(data_list)
                        data_parameters = {'Name': data_name,
                                           'format': self._format,
                                           'RangeMin': min_val,
                                           'RangeMax': max_val}
                        xml.write_cell_data_array(vtk_file, data_list,
                                                  data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Von Mises equivalent stress
        array_vox = voxels_array_factory.build_voxels_array(
            crve, 'vm_stress', clusters_state, clusters_def_gradient_mf)[0]
        data_list = list(array_vox.flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': 'Von Mises Eq. Stress',
                           'format': self._format,
                           'RangeMin': min_val,
                           'RangeMax': max_val}
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Von Mises equivalent strain
        array_vox = voxels_array_factory.build_voxels_array(
            crve, 'vm_strain', clusters_state, clusters_def_gradient_mf)[0]
        data_list = list(array_vox.flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': 'Von Mises Eq. Strain',
                           'format': self._format,
                           'RangeMin': min_val,
                           'RangeMax': max_val}
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK cell data array - Cluster adaptive level
        if adaptivity_manager is not None:
            rg_array = adaptivity_manager.get_adapt_vtk_array(voxels_clusters)
            data_list = list(rg_array.flatten('F'))
            min_val = min(data_list)
            max_val = max(data_list)
            data_parameters = {'Name': 'Adaptive level',
                               'format': self._format,
                               'RangeMin': min_val,
                               'RangeMax': max_val}
            xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element cell data
        xml.write_close_cell_data(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element piece
        xml.write_close_dataset_piece(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element
        xml.write_close_dataset_elem(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file footer
        xml.write_file_footer(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close clustering VTK file
        vtk_file.close()
    # -------------------------------------------------------------------------
    def rewind_files(self, rewind_inc):
        """Rewind VTK output files.

        Parameters
        ----------
        rewind_inc : int
            Increment associated with the rewind state.
        """
        # Get VTK output files
        vtk_files_paths = [os.path.join(self._vtk_dir, file_path)
                           for file_path in os.listdir(self._vtk_dir)
                           if os.path.isfile(os.path.join(self._vtk_dir,
                                                          file_path))]
        # Loop over VTK output files
        for vtk_file_path in vtk_files_paths:
            # Get VTK output file time step
            time_step = int(os.path.splitext(vtk_file_path)[0].split('_')[-1])
            # Delete VTK output file and remove it from VTK collection
            if time_step > rewind_inc:
                # Delete VTK output file
                os.remove(vtk_file_path)
                # Remove VTK output file from VTK collection
                self._vtk_collection.remove_vtk_collection_file(
                    time_step_file_path='VTK/' + vtk_file_path.split('VTK/',
                                                                     1)[1])
    # -------------------------------------------------------------------------
    @staticmethod
    def _set_image_data_parameters(rve_dims, n_voxels_dims):
        """Set ImageData dataset parameters.

        Parameters
        ----------
        rve_dims : list[float]
            RVE size in each dimension.
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).

        Returns
        -------
        dataset_parameters : dict
            ImageData dataset parameters.
        dataset_parameters : dict
            ImageData dataset piece parameters.
        """
        # Set WholeExtent parameter
        whole_extent = list(copy.deepcopy(n_voxels_dims))
        for i in range(len(whole_extent)):
            whole_extent.insert(2*i, 0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Origin parameter
        origin = [0, 0, 0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Spacing parameter
        spacing = [rve_dims[i]/n_voxels_dims[i] for i in range(len(rve_dims))]
        # Set null third dimension in 2D problem
        if len(whole_extent) == 4:
            whole_extent = whole_extent + [0, 1]
            spacing.append(0.0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build ImageData dataset parameters
        dataset_parameters = {'WholeExtent': whole_extent, 'Origin': origin,
                              'Spacing': spacing}
        # Build ImageData dataset piece parameters
        piece_parameters = {'Extent': whole_extent}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return dataset_parameters, piece_parameters
    # -------------------------------------------------------------------------
    def _set_cell_data_name(self, comp_order_sym, comp_order_nsym, var_name,
                            var_type, idx=None, model_name=None):
        """Set variable cell data array name.

        Parameters
        ----------
        comp_order_sym : list[str]
            Strain/Stress components (str) order (symmetric).
        comp_order_nsym : list[str]
            Strain/Stress components (str) order (non-symmetric).
        var_name : str
            Variable name.
        var_type : str
            Variable type.
        idx : str, default=None
            Component index.
        model_name : str, default=None
            Material constitutive model name. If provided, the model name is
            included as a prefix in the variable cell data array name.
        """
        # Set cell data array name prefix
        if model_name is not None:
            prefix = model_name + ': '
        else:
            prefix = ''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output variable name
        if var_type in ['int', 'bool', 'float']:
            data_name = prefix + var_name
        elif var_type == 'vector':
            data_name = prefix + var_name + '_' + str(idx)
        elif var_type == 'sym_matrix_mf':
            data_name = prefix + var_name[:-3] + '_' + comp_order_sym[idx]
        elif var_type == 'nsym_matrix_mf':
            data_name = prefix + var_name[:-3] + '_' + comp_order_nsym[idx]
        else:
            data_name = prefix + var_name + '_' + comp_order_nsym[idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return data_name
    # -------------------------------------------------------------------------
    def _set_state_var_descriptors(
            self, n_dim, comp_order_sym, comp_order_nsym, var_name, source,
            model_state_variables=None, cluster=None, clusters_state=None):
        """Set state variable descriptors required to output cell data array.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order_sym : list[str]
            Strain/Stress components (str) order (symmetric).
        comp_order_nsym : list[str]
            Strain/Stress components (str) order (non-symmetric).
        var_name : str
            State variable name.
        source : {'model_state_variables', 'clusters_state'}
            Source where state variable is stored.
        model_state_variables : dict, default=None
            Material model state variables (required if
            `source`='model_state_variables').
        cluster : int, default=None
            Cluster label (required if `source`='clusters_state').
        clusters_state : dict, default=None
            Material constitutive model state variables (item, dict) associated
            with each material cluster (key, str) (required if
            `source`='clusters_state').

        Returns
        -------
        var : {int, float, bool,  array}
            State variable.
        var_type : str
            State variable type.
        var_n_comp : int
            State variable number of components.
        """
        # Get stored state variable for output
        if source == 'model_state_variables':
            if model_state_variables is None:
                raise RuntimeError('State variable source is not available.')
            else:
                stored_var = model_state_variables[var_name]
        elif source == 'clusters_state':
            if cluster is None or clusters_state is None:
                raise RuntimeError('State variable source is not available.')
            else:
                stored_var = clusters_state[str(cluster)][var_name]
        else:
            raise RuntimeError('Unknown source of state variable.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output state variable descriptors
        if isinstance(stored_var, int) or isinstance(stored_var, np.integer):
            var_type = 'int'
            var_n_comp = 1
            var = stored_var
        elif isinstance(stored_var, float) or isinstance(stored_var, np.float):
            var_type = 'float'
            var_n_comp = 1
            var = stored_var
        elif isinstance(stored_var, bool):
            var_type = 'bool'
            var_n_comp = 1
            var = stored_var
        elif isinstance(stored_var, np.ndarray) and len(stored_var.shape) == 1:
            if var_name.split('_')[-1] == 'mf':
                if len(stored_var) == len(comp_order_sym):
                    var_type = 'sym_matrix_mf'
                    var_n_comp = len(comp_order_sym)
                    var = mop.get_tensor_from_mf(stored_var, n_dim,
                                                 comp_order_sym)
                else:
                    var_type = 'nsym_matrix_mf'
                    var_n_comp = len(comp_order_nsym)
                    var = mop.get_tensor_from_mf(stored_var, n_dim,
                                                 comp_order_nsym)
        else:
            var_type = 'matrix'
            var_n_comp = len(comp_order_nsym)
            var = stored_var
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return var, var_type, var_n_comp
    # -------------------------------------------------------------------------
    def _state_var_cell_data_array(
            self, n_dim, comp_order_sym, comp_order_nsym, material_phases,
            material_phases_models, phase_clusters, voxels_clusters,
            clusters_state, var_name, var_type, comp_idx=None,
            model_name=None):
        """Build state variable cell data array.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order_sym : list[str]
            Strain/Stress components (str) order (symmetric).
        comp_order_nsym : list[str]
            Strain/Stress components (str) order (non-symmetric).
        material_phases : list[str]
            CRVE material phases labels (str).
        material_phases_models : dict
            Material constitutive model (item, ConstitutiveModel) associated
            with each material phase (key, str).
        phase_clusters : dict
            Clusters labels (item, list of int) associated with each material
            phase (key, str).
        voxels_clusters : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the cluster label (int) assigned to the
            corresponding voxel.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            with each material cluster (key, str) (required if
            `source`='clusters_state').
        var_name : str
            State variable name.
        var_type : str
            State variable type.
        comp_idx : str, default=None
            Tensorial component index.
        model_name : str, default=None
            Material constitutive model name. If not provided, it is assumed
            that the state variable is common to all material phases and
            associated constitutive models.

        Returns
        -------
        rg_array : numpy.ndarray (2d or 3d)
            Array of state variable component of shape equal to RVE regular
            grid discretization array.
        """
        # Initialize regular grid shape array
        rg_array = copy.deepcopy(voxels_clusters)
        rg_array = rg_array.astype(str)
        rg_array = rg_array.astype(object)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in material_phases:
            # Get material phase constitutive model
            constitutive_model = material_phases_models[str(mat_phase)]
            # Build state variable cell data array
            if model_name is None \
                    or constitutive_model.get_name() == model_name:
                # Loop over material phase clusters
                for cluster in phase_clusters[mat_phase]:
                    # Get material cluster state variable
                    cluster_var, _, _ = self._set_state_var_descriptors(
                        n_dim, comp_order_sym, comp_order_nsym, var_name,
                        source='clusters_state', cluster=cluster,
                        clusters_state=clusters_state)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble material cluster state variable
                    if var_type in ['int', 'bool', 'float']:
                        rg_array = np.where(rg_array == str(cluster),
                                            cluster_var, rg_array)
                    elif var_type == 'vector':
                        rg_array = np.where(rg_array == str(cluster),
                                            cluster_var[comp_idx], rg_array)
                    elif var_type == 'sym_matrix_mf':
                        idx = tuple([int(x) - 1
                                     for x in comp_order_sym[comp_idx]])
                        rg_array = np.where(rg_array == str(cluster),
                                            cluster_var[idx], rg_array)
                    elif var_type == 'nsym_matrix_mf':
                        idx = tuple([int(x) - 1
                                     for x in comp_order_nsym[comp_idx]])
                        rg_array = np.where(rg_array == str(cluster),
                                            cluster_var[idx], rg_array)
                    else:
                        idx = tuple([int(x) - 1
                                     for x in comp_order_nsym[comp_idx]])
                        rg_array = np.where(rg_array == str(cluster),
                                            cluster_var[idx], rg_array)
            else:
                # Loop over material phase clusters
                for cluster in phase_clusters[mat_phase]:
                    # Assemble a default state variable value for all clusters
                    # for which the associated material phase is not governed
                    # by the provided material constitutive model
                    if var_type in ['int']:
                        rg_array = np.where(rg_array == str(cluster), int(0),
                                            rg_array)
                    elif var_type in ['bool']:
                        rg_array = np.where(rg_array == str(cluster), False,
                                            rg_array)
                    else:
                        rg_array = np.where(rg_array == str(cluster), float(0),
                                            rg_array)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if the state variable has been specified for every pixels
        # (2D) / voxels (3D)
        # in order to build a valid output cell data array
        if any(isinstance(x, str) for x in list(rg_array.flatten('F'))):
            raise RuntimeError('Incomplete VTK output cell data array - '
                               + var_name + '.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return rg_array
    # -------------------------------------------------------------------------
    @staticmethod
    def _reset_labels_from_zero(array_1d):
        """Reset 1d array of integers starting from zero.

        Parameters
        ----------
        array_1d : numpy.ndarray (1d)
            1d array of integers.
        Returns
        -------
        array_1d_new : numpy.ndarray (1d)
            Relabeled 1d array of integers.
        """
        # Get sorted old labels
        old_labels = np.array(sorted(set(array_1d)))
        # Set new labels
        new_labels = np.array(range(len(set(array_1d))))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize relabeled 1d array
        array_1d_new = np.full(len(array_1d), -1, dtype=int)
        # Loop over old labels
        for i in range(len(old_labels)):
            # Get old label indexes
            idxs = np.where(array_1d == old_labels[i])
            # Reset old label
            array_1d_new[idxs] = new_labels[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all labels have been reset successfuly
        if np.any(array_1d_new == -1):
            raise RuntimeError('At least one label has not been successfuly '
                               'reset.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return array_1d_new
# =============================================================================
class VTKCollection:
    """VTK collection.

    Attributes
    ----------
    _byte_order : str
        VTK byte order.
    _indent : str
        Formatting indentation.

    Methods
    -------
    init_vtk_collection_file(self)
        Open VTK collection output file and write file header and footer.
    write_vtk_collection_file(self, time_step, time_step_file_path)
        Add VTK file to VTK collection file.
    remove_vtk_collection_file(self, time_step_file_path)
        Remove VTK file from VTK collection file.
    """
    def __init__(self, pvd_file_path):
        """VTK collection constructor.

        Parameters
        ----------
        pvd_file_path : str
            Path of VTK collection output file.
        """
        self._pvd_file_path = pvd_file_path
        # Set byte order
        if sys.byteorder == 'little':
            self._byte_order = 'LittleEndian'
        else:
            self._byte_order = 'BigEndian'
        # Set formatting indentation
        self._indent = '  '
    # -------------------------------------------------------------------------
    def init_vtk_collection_file(self):
        """Open VTK collection output file and write file header and footer."""
        # Open VTK collection file (append mode)
        vtk_file = open(self._pvd_file_path, 'a')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK collection file header
        vtk_file.write('<' + '?xml version="1.0"?' + '>' + '\n')
        vtk_file.write('<' + 'VTKFile type='
                       + XMLGenerator.enclose('Collection') + ' '
                       + 'version=' + XMLGenerator.enclose('0.1') + ' '
                       + 'byte_order=' + XMLGenerator.enclose(self._byte_order)
                       + '>' + '\n')
        # Open VTK collection element
        vtk_file.write(self._indent + '<' + 'Collection' + '>' + '\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK collection element
        vtk_file.write(self._indent + '<' + '/Collection' + '>' + '\n')
        # Close VTK collection file
        vtk_file.write('<' + '/VTKFile' + '>' + '\n')
        # Close VTK collection file
        vtk_file.close()
    # -------------------------------------------------------------------------
    def write_vtk_collection_file(self, time_step, time_step_file_path):
        """Add VTK file to VTK collection file.

        Parameters
        ----------
        time_step : int
            VTK file time step.
        time_step_file_path : str
            Path of time step VTK file.
        """
        # Open VTK collection file and read lines (read)
        file_lines = open(self._pvd_file_path, 'r').readlines()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add time step VTK file
        file_lines.insert(-2, 2*self._indent + '<' + 'DataSet' + ' '
                          + 'timestep=' + XMLGenerator.enclose(time_step) + ' '
                          + 'file=' + XMLGenerator.enclose(time_step_file_path)
                          + '/>' + '\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write updated VTK collection file
        open(self._pvd_file_path, 'w').writelines(file_lines)
    # -------------------------------------------------------------------------
    def remove_vtk_collection_file(self, time_step_file_path):
        """Remove VTK file from VTK collection file.

        Parameters
        ----------
        time_step_file_path : str
            Path of time step VTK file.
        """
        # Open VTK collection file and read lines (read)
        file_lines = open(self._pvd_file_path, 'r').readlines()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over VTK output files
        for i in range(len(file_lines)):
            # Find index of VTK output file to be removed
            if time_step_file_path in file_lines[i]:
                break
            # Raise error if VTK output file to be removed does not exist
            if i == len(file_lines) - 1:
                raise RuntimeError('VTK output file to be removed does not '
                                   'exist.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove VTK output file
        file_lines.pop(i)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write updated VTK collection file
        open(self._pvd_file_path, 'w').writelines(file_lines)
# =============================================================================
class XMLGenerator:
    """VTK XML files generation methods.

    Attributes
    ----------
    _indent : str
        Formatting indentation.

    Methods
    -------
    write_file_header(self, vtk_file)
        Write VTK file header.
    write_file_footer(self, vtk_file)
        Write VTK file footer.
    write_open_dataset_elem(self, vtk_file, dataset_parameters)
        Open (write) VTK dataset element.
    write_close_dataset_elem(self, vtk_file)
        Close (write) VTK dataset element.
    write_open_dataset_piece(self, vtk_file, piece_parameters)
        Open (write) VTK dataset element piece.
    write_close_dataset_piece(self, vtk_file)
        Close (write) VTK dataset element piece.
    write_open_point_data(self, vtk_file)
        Open (write) VTK point data element.
    write_close_point_data(self, vtk_file)
        Close (write) VTK point data element.
    write_open_cell_data(self, vtk_file)
        Open (write) VTK cell data element.
    write_close_cell_data(self, vtk_file)
        Close (write) VTK cell data element.
    write_cell_data_array(self, vtk_file, data_list, data_parameters)
        Write VTK cell data element.
    enclose(x)
        Enclose input in literal quotation marks.
    """
    def __init__(self, type, version, byte_order, format, precision,
                 header_type):
        """VTK XML file constructor.

        Parameters
        ----------
        type : str
            VTK file type.
        version : str
            VTK version.
        byte_order : str
            VTK byte order.
        format : str
            VTK file format.
        precision : {'SinglePrecision', 'DoublePrecision'}
            VTK file data precision.
        header_type : str
            VTK header type.
        """
        self._type = type
        self._version = version
        self._byte_order = byte_order
        self._format = format
        self._precision = precision
        self._header_type = header_type
        # Set formatting indentation
        self._indent = '  '
    # -------------------------------------------------------------------------
    def write_file_header(self, vtk_file):
        """Write VTK file header.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write('<' + '?xml version="1.0"?' + '>' + '\n')
        vtk_file.write('<' + 'VTKFile type=' + XMLGenerator.enclose(self._type)
                       + ' ' + 'version=' + XMLGenerator.enclose(self._version)
                       + ' ' + 'byte_order='
                       + XMLGenerator.enclose(self._byte_order) + ' '
                       + 'header_type='
                       + XMLGenerator.enclose(self._header_type) + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_file_footer(self, vtk_file):
        """Write VTK file footer.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write('<' + '/VTKFile' + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_open_dataset_elem(self, vtk_file, dataset_parameters):
        """Open (write) VTK dataset element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        dataset_parameters : dict
            VTK dataset element parameters.
        """
        parameters = copy.deepcopy(dataset_parameters)
        vtk_file.write(self._indent + '<' + self._type + ' '
                       + ' '.join([key + '='
                                   + XMLGenerator.enclose(parameters[key])
                                   for key in parameters]) + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_close_dataset_elem(self, vtk_file):
        """Close (write) VTK dataset element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write(self._indent + '<' + '/' + self._type + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_open_dataset_piece(self, vtk_file, piece_parameters):
        """Open (write) VTK dataset element piece.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        piece_parameters : dict
            VTK dataset element piece parameters.
        """
        parameters = copy.deepcopy(piece_parameters)
        vtk_file.write(2*self._indent + '<' + 'Piece' + ' '
                       + ' '.join([key + '='
                                   + XMLGenerator.enclose(parameters[key])
                                   for key in parameters]) + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_close_dataset_piece(self, vtk_file):
        """Close (write) VTK dataset element piece.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write(2*self._indent + '</Piece>' + '\n')
    # -------------------------------------------------------------------------
    def write_open_point_data(self, vtk_file):
        """Open (write) VTK point data element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write(3*self._indent + '<' + 'PointData' + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_close_point_data(self, vtk_file):
        """Close (write) VTK point data element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write(3*self._indent + '<' + '/PointData' + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_open_cell_data(self, vtk_file):
        """Open (write) VTK cell data element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write(3*self._indent + '<' + 'CellData' + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_close_cell_data(self, vtk_file):
        """Close (write) VTK cell data element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        """
        vtk_file.write(3*self._indent + '<' + '/CellData' + '>' + '\n')
    # -------------------------------------------------------------------------
    def write_cell_data_array(self, vtk_file, data_list, data_parameters):
        """Write VTK cell data element.

        Parameters
        ----------
        vtk_file : file
            VTK file.
        data_list : list
            Sorted data of cell element.
        data_parameters : dict
            Parameters of cell element.
        """
        # Set cell data array data type and associated ascii format
        if all(isinstance(x, int) or isinstance(x, np.integer)
                for x in data_list):
            data_type = 'Int32'
            frmt = 'd'
        elif all('bool' in str(type(x)).lower() for x in data_list):
            data_type = 'Int32'
            frmt = 'd'
        else:
            if self._precision == 'SinglePrecision':
                data_type = 'Float32'
                frmt = '16.8e'
            elif self._precision == 'DoublePrecision':
                data_type = 'Float64'
                frmt = '25.16e'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize cell data array parameters
        parameters = dict()
        values = data_list
        # Set cell data array name
        if 'Name' in data_parameters.keys():
            parameters['Name'] = data_parameters['Name']
        else:
            parameters['Name'] = '?'
        # Set cell data array type
        parameters['type'] = data_type
        # Set cell data array number of components
        if 'NumberofComponents' in data_parameters.keys():
            parameters['NumberofComponents'] = \
                data_parameters['NumberofComponents']
        # Set cell data array format
        if 'format' in data_parameters.keys():
            parameters['format'] = data_parameters['format']
        else:
            raise RuntimeError('Unknown format.')
        # Set cell data array range
        if 'RangeMin' in data_parameters.keys():
            parameters['RangeMin'] = data_parameters['RangeMin']
            min_val = data_parameters['RangeMin']
            # Mask data values according to specified range lower bound
            values = list(np.where(np.array(values) < min_val, min_val,
                                   np.array(values)))
        if 'RangeMax' in data_parameters.keys():
            parameters['RangeMax'] = data_parameters['RangeMax']
            max_val = data_parameters['RangeMax']
            # Mask data values according to specified range upper bound
            values = list(np.where(np.array(values) > max_val, max_val,
                                   np.array(values)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK data array header
        vtk_file.write(4*self._indent + '<' + 'DataArray' + ' '
                       + ' '.join([key + '='
                                   + XMLGenerator.enclose(parameters[key])
                                   for key in parameters]) + '>' + '\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK data array values
        n_line_vals = 6
        template1 = 5*self._indent + n_line_vals*('{: ' + frmt + '}') + '\n'
        template2 = 5*self._indent \
            + (len(values) % n_line_vals)*('{: ' + frmt + '}') + '\n'
        aux_list = [values[i:i+n_line_vals]
                    for i in range(0, len(values), n_line_vals)]
        for i in range(len(aux_list)):
            if i == len(aux_list) - 1 and len(values) % n_line_vals != 0:
                vtk_file.write(template2.format(*aux_list[i]))
            else:
                vtk_file.write(template1.format(*aux_list[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK data array footer
        vtk_file.write(4*self._indent + '<' + '/DataArray' + '>' + '\n')
    # -------------------------------------------------------------------------
    @staticmethod
    def enclose(x):
        """Enclose input in literal quotation marks.

        Parameters
        ----------
        x : str
            Input to be converted into a string enclosed in literal quotation
            marks.

        Returns
        -------
        str : str
            Input converted into a string enclosed in literal quotation marks.
        """
        if isinstance(x, str):
            return '\'' + x + '\''
        elif isinstance(x, list):
            return '\'' + ' '.join(str(i) for i in x) + '\''
        else:
            return '\'' + str(x) + '\''
