#
# Links Finite Element Method-based Offline-Stage (CRATE Program)
# ==========================================================================================
# Summary:
# Class that performs the required DNS simulations of the CROM offline-stage through the
# FEM-based first-order multi-scale hierarchical model based on computational homogenization
# implemented in the multi-scale finite element code Links (Large Strain Implicit Nonlinear
# Analysis of Solids Linking Scales), developed by the CM2S research group at the Faculty of
# Engineering, University of Porto.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Nov 2021 | Refactoring and OOP implementation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Subprocess management
import subprocess
# Extract information from path
import ntpath
# Working with arrays
import numpy as np
# Manage files and directories
import ioput.fileoperations as filop
# Matricial operations
import tensor.matrixoperations as mop
# DNS homogenization methods interface
from clustering.solution.dnshomogenization import DNSHomogenizationMethod
# Links configuration
from links.configuration import get_links_analysis_type, get_links_comp_order
# Links constitutive models
from links.material.models.links_elastic import LinksElastic
#
#                                                      Links FEM-based homogenization method
# ==========================================================================================
class LinksFEMHomogenization(DNSHomogenizationMethod):
    '''Links FEM-based homogenization method.

    FEM-based first-order multi-scale hierarchical model based on computational
    homogenization implemented in multi-scale finite element code Links (Large Strain
    Implicit Nonlinear Analysis of Solids Linking Scales), developed by the CM2S research
    group at the Faculty of Engineering, University of Porto.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _analysis_type : int
        Links analysis type: 2D plane stress (1), 2D plane strains (2), 2D axisymmetric (3)
        and 3D (6).
    _n_material_phases : int
        Number of material phases.
    '''
    def __init__(self, strain_formulation, problem_type, rve_dims, n_voxels_dims,
                 regular_grid, material_phases, material_phases_properties, links_bin_path,
                 links_offline_dir, fe_order='quadratic',
                 boundary_type='Periodic_Condition', convergence_tolerance=1e-6,
                 element_avg_output_mode=1):
        '''Links FEM-based homogenization method constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        rve_dims : list
            RVE size in each dimension.
        n_voxels_dims : list
            Number of voxels in each dimension of the regular grid (spatial discretization
            of the RVE).
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to each material
            phase (key, str).
        links_bin_path : str
            Links binary absolute path.
        links_offline_dir : str
            Directory where Links offline-stage simulation files are stored.
        fe_order : str, {'linear', 'quadratic'}, default='quadratic'
            Finite element order of quadrilateral (2D) or hexahedral (3D) elements.
        boundary_type : str, {'Taylor_Condition', 'Linear_Condition', 'Periodic_Condition',
                              'Uniform_Traction_Condition', 'Uniform_Traction_Condition_II',
                              'Mortar_Periodic_Condition', 'Mortar_Periodic_Condition_LM'},
                             default='Periodic_Condition'
            Microscale boundary condition.
        convergence_tolerance : float, default=1e-6
            Convergence tolerance.
        element_avg_output_mode : int, {1,}, default=1
            Element average output mode: infinitesimal strain tensor (infinitesimal
            strains) / material logarithmic strain tensor (finite strains) (1).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._regular_grid = regular_grid
        self._material_phases = material_phases
        self._material_phases_properties = material_phases_properties
        self._links_bin_path = links_bin_path
        self._links_offline_dir = links_offline_dir
        self._fe_order = fe_order
        self._boundary_type = boundary_type
        self._convergence_tolerance = convergence_tolerance
        self._element_avg_output_mode = element_avg_output_mode
        # Get problem type parameters
        n_dim, comp_order_sym, _ = mop.get_problem_type_parameters(self._problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        # Set Links analysis type
        self._analysis_type = get_links_analysis_type(problem_type)
        # Get number of material phases
        self._n_material_phases = len(self._material_phases)
    # --------------------------------------------------------------------------------------
    def compute_rve_local_response(self, mac_strain_id, mac_strain):
        '''Compute RVE local elastic strain response.

        Parameters
        ----------
        mac_strain_id : int
            Macroscale strain second-order tensor identifier.
        mac_strain : 2darray
            Macroscale strain second-order tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).

        Returns
        -------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str). Infinitesimal strain
            tensor (infinitesimal strains) or material logarithmic strain tensor (finite
            strains).
        '''
        # Set Links input file name
        links_file_name = 'mac_strain_' + str(mac_strain_id)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate Links microscale  input data file
        links_file_path = self._write_links_input_data_file(links_file_name, mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve RVE microscale equilibrium problem
        self._run_links(links_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE local elastic strain response from Links output file
        strain_vox = self._get_strain_vox(links_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_vox
    # --------------------------------------------------------------------------------------
    def _write_links_input_data_file(self, file_name, mac_strain):
        '''Generate Links microscale input data file.

        Parameters
        ----------
        file_name : str
            Links input data file name.
        mac_strain : 2darray
            Macroscale strain second-order tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).

        Returns
        -------
        file_path : str
            Links input data file path.
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set and create Links offline-stage directory if it does not exist
        if not os.path.exists(self._links_offline_dir):
            filop.makedirectory(self._links_offline_dir)
        # Set Links input data file path
        links_file_path = self._links_offline_dir + file_name + '.rve'
        # Abort if attempting to overwrite an existing Links input data file
        if os.path.isfile(links_file_path):
            raise RuntimeError('Attempt to overwrite an existing Links input data file.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links input data file fixed parameters:
        title = 'Links input data file generated automatically by CRATE program'
        if self._strain_formulation == 'infinitesimal':
            # Set finite strains flag
            large_strain_formulation = 'OFF'
            # Set strain prescription keyword
            strain_keyword = 'Prescribed_Epsilon'
            # Set number of loading increments
            number_of_increments = 1
        else:
            # Set finite strains flag
            large_strain_formulation = 'ON'
            # Set strain prescription keyword
            strain_keyword = 'Prescribed_Deformation_Gradient'
            # Set number of loading increments
            number_of_increments = 1
        # Set solver and associated parallelization
        solver = 'PARDISO'
        parallel_solver = 4
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate Links finite element mesh
        coords, connectivities, element_phases = self._generate_links_fe_mesh()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create a file containing solely the Links finite element mesh data if it does not
        # exist yet
        mesh_path = self._links_offline_dir + 'rgmsh_conversion' + '.femsh'
        if not os.path.isfile(mesh_path):
            self._write_links_mesh_data(mesh_path, coords, connectivities, element_phases)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open Links input data file
        links_file = open(links_file_path, 'w')
        # Format file structure
        write_list = ['\n' + 'TITLE ' + '\n' + title + '\n'] + \
                     ['\n' + 'ANALYSIS_TYPE ' + str(self._analysis_type) + '\n'] + \
                     ['\n' + 'LARGE_STRAIN_FORMULATION ' + large_strain_formulation +
                      '\n'] + \
                     ['\n' + 'Boundary_Type ' + self._boundary_type + '\n'] + \
                     ['\n' + strain_keyword + '\n'] + \
                     [' '.join([str('{:16.8e}'.format(mac_strain[i, j]))
                      for j in range(self._n_dim)]) + '\n' for i in range(self._n_dim)] + \
                     ['\n'] + \
                     ['Number_of_Increments ' + str(number_of_increments) + '\n'] + \
                     ['\n' + 'CONVERGENCE_TOLERANCE' + '\n' +
                      str(self._convergence_tolerance) + '\n'] + \
                     ['\n' + 'SOLVER ' + solver + '\n'] + \
                     ['\n' + 'PARALLEL_SOLVER ' + str(parallel_solver) + '\n'] + \
                     ['\n' + 'VTK_OUTPUT NONE' + '\n'] + \
                     ['\n' + 'Element_Average_Output ' +
                      str(self._element_avg_output_mode) + '\n'] + \
                     ['\n' + 'MESH_FILE ' + mesh_path + '\n'] + \
                     ['\n' + 'MATERIALS ' + str(self._n_material_phases) + '\n']
        # Write Links input data file
        links_file.writelines(write_list)
        # Close Links input data file
        links_file.close()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Instantiate elastic constitutive model
            constitutive_model = LinksElastic(self._strain_formulation, self._problem_type,
                                              self._material_phases_properties[mat_phase])
            # Append constitutive model elastic properties
            constitutive_model.write_mat_properties(links_file_path, mat_phase)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return links_file_path
    # --------------------------------------------------------------------------------------
    def _write_links_mesh_data(self, file_path, coords, connectivities, element_phases):
        '''Append Links material and finite element mesh data to input data file.

        Parameters
        ----------
        file_path : str
            Links input data file path.
        coords : dict
            Spatial coordinates (item, list of float) associated to each node (key, str).
        connectivities : dict
            Nodes (item, list of int) associated to each finite element (key, str).
        element_phases : dict
            Material phase (item, int) associated to each finite element (key, str).
        '''
        # Set element designation and number of Gauss integration points
        if self._n_dim == 2:
            if self._fe_order == 'linear':
                elem_type = 'QUAD4'
                n_gp = 4
            else:
                elem_type = 'QUAD8'
                n_gp = 4
        else:
            if self._fe_order == 'linear':
                elem_type = 'HEXA8'
                n_gp = 8
            else:
                elem_type = 'HEXA20'
                n_gp = 8
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open data file to append Links finite element mesh
        data_file = open(file_path, 'a')
        # Format file structure
        write_list = ['\n' + 'ELEMENT_GROUPS ' + str(self._n_material_phases) + '\n'] + \
                     [str(mat+1) + ' 1 ' + str(mat+1) + '\n' \
                      for mat in range(self._n_material_phases)] + \
                     ['\n' + 'ELEMENT_TYPES 1' + '\n', '1 ' + elem_type + '\n', '  ' + \
                      str(n_gp) + ' GP' + '\n'] + \
                     ['\n' + 'MATERIALS ' + str(self._n_material_phases) + '\n']
        # Append first part of the Links finite element mesh to data file
        data_file.writelines(write_list)
        # Close data file
        data_file.close()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Instantiate elastic constitutive model
            constitutive_model = LinksElastic(self._strain_formulation, self._problem_type,
                                              self._material_phases_properties[mat_phase])
            # Append constitutive model elastic properties
            constitutive_model.write_mat_properties(file_path, mat_phase)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open data file to append material and finite element mesh data
        data_file = open(file_path, 'a')
        # Format file structure
        write_list = ['\n' + 'ELEMENTS ' + str(len(connectivities.keys())) + '\n'] + \
                     ['{:>3s}'.format(str(elem)) + \
                      '{:^5d}'.format(element_phases[str(elem)]) + ' '.join([str(node) \
                      for node in connectivities[str(elem)]]) + '\n' \
                      for elem in np.sort([int(key) for key in connectivities.keys()])] + \
                     ['\n' + 'NODE_COORDINATES ' + str(len(coords.keys())) + \
                      ' CARTESIAN' + '\n'] + \
                     ['{:>3s}'.format(str(node)) + ' ' + \
                      ' '.join([str('{:16.8e}'.format(coord)) \
                      for coord in coords[str(node)]]) + '\n' \
                      for node in np.sort([int(key) for key in coords.keys()])]
        # Append last part of the Links finite element mesh to data file
        data_file.writelines(write_list)
        # Close data file
        data_file.close()
    # --------------------------------------------------------------------------------------
    def _generate_links_fe_mesh(self):
        '''Generate Links regular finite element mesh.

        Returns
        -------
        coords : dict
            Spatial coordinates (item, list of float) associated to each node (key, str).
        connectivities : dict
            Nodes (item, list of int) associated to each finite element (key, str).
        element_phases : dict
            Material phase (item, int) associated to each finite element (key, str).
        '''
        # Initialize array with finite element mesh nodes
        if self._fe_order == 'linear':
            nodes_grid = np.zeros(np.array(self._n_voxels_dims) + 1, dtype=int)
        else:
            nodes_grid = np.zeros(2*np.array(self._n_voxels_dims) + 1, dtype=int)
        # Initialize coordinates dictionary
        coords = dict()
        # Initialize connectivities dictionary
        connectivities = dict()
        # Initialize element phases dictionary
        element_phases = dict()
        # Set sampling periods in each dimension
        sampling_period = [self._rve_dims[i]/self._n_voxels_dims[i]
                           for i in range(self._n_dim)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set nodes coordinates
        node = 1
        if self._n_dim == 2:
            # Set nodes for linear (QUAD4) or quadratic (QUAD8) finite element mesh
            if self._fe_order == 'linear':
                # Loop over nodes
                for j in range(self._n_voxels_dims[1] + 1):
                    for i in range(self._n_voxels_dims[0] + 1):
                        nodes_grid[i, j] = node
                        # Set node coordinates
                        coords[str(node)] = [i*sampling_period[0], j*sampling_period[1]]
                        # Increment node counter
                        node = node + 1
            elif self._fe_order == 'quadratic':
                # Loop over nodes
                for j in range(2*self._n_voxels_dims[1] + 1):
                    for i in range(2*self._n_voxels_dims[0] + 1):
                        if j % 2 != 0 and i % 2 != 0:
                            # Skip inexistent node
                            nodes_grid[i, j] = -1
                        else:
                            nodes_grid[i, j] = node
                            # Set node coordinates
                            coords[str(node)] = [i*0.5*sampling_period[0],
                                                 j*0.5*sampling_period[1]]
                            # Increment node counter
                            node = node + 1
        elif self._n_dim == 3:
            # Set nodes for linear (HEXA8) or quadratic (HEXA20) finite element mesh
            if self._fe_order == 'linear':
                # Loop over nodes
                for k in range(self._n_voxels_dims[2]+1):
                    for j in range(self._n_voxels_dims[1]+1):
                        for i in range(self._n_voxels_dims[0]+1):
                            nodes_grid[i, j, k] = node
                            # Set node coordinates
                            coords[str(node)] = [i*sampling_period[0],
                                                 j*sampling_period[1],
                                                 k*sampling_period[2]]
                            # Increment node counter
                            node = node + 1
            if self._fe_order == 'quadratic':
                # Loop over nodes
                for k in range(2*self._n_voxels_dims[2] + 1):
                    for j in range(2*self._n_voxels_dims[1] + 1):
                        for i in range(2*self._n_voxels_dims[0] + 1):
                            # Skip inexistent node
                            if (j % 2 != 0 and i % 2 != 0) or \
                                    (k % 2 != 0 and (j % 2 != 0 or i % 2 != 0)):
                                nodes_grid[i, j, k] = -1
                            else:
                                # Set node coordinates
                                nodes_grid[i, j, k] = node
                                coords[str(node)] = [i*0.5*sampling_period[0],
                                                     j*0.5*sampling_period[1],
                                                     k*0.5*sampling_period[2]]
                                # Increment node counter
                                node = node + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set element connectivities and material phases
        elem = 1
        if self._n_dim == 2:
            # Set linear (QUAD4) or quadratic (QUAD8) finite element mesh connectivities
            if self._fe_order == 'linear':
                # Loop over elements
                for j in range(self._n_voxels_dims[1]):
                    for i in range(self._n_voxels_dims[0]):
                        # Set element connectivities
                        connectivities[str(elem)] = [nodes_grid[i, j], nodes_grid[i + 1, j],
                                                     nodes_grid[i + 1, j + 1],
                                                     nodes_grid[i, j + 1]]
                        # Set element material phase
                        element_phases[str(elem)] = self._regular_grid[i, j]
                        # Increment element counter
                        elem = elem + 1
            elif self._fe_order == 'quadratic':
                # Loop over elements
                for j in range(self._n_voxels_dims[1]):
                    for i in range(self._n_voxels_dims[0]):
                        # Set element connectivities
                        connectivities[str(elem)] = [nodes_grid[2*i, 2*j],
                                                     nodes_grid[2*i + 1, 2*j],
                                                     nodes_grid[2*i + 2, 2*j],
                                                     nodes_grid[2*i + 2, 2*j + 1],
                                                     nodes_grid[2*i + 2, 2*j + 2],
                                                     nodes_grid[2*i + 1, 2*j + 2],
                                                     nodes_grid[2*i, 2*j + 2],
                                                     nodes_grid[2*i, 2*j + 1]]
                        # Set element material phase
                        element_phases[str(elem)] = self._regular_grid[i, j]
                        # Increment element counter
                        elem = elem + 1
        elif self._n_dim == 3:
            # Set linear (HEXA8) or quadratic (HEXA20) finite element mesh connectivities
            if self._fe_order == 'linear':
                # Loop over elements
                for k in range(self._n_voxels_dims[2]):
                    for j in range(self._n_voxels_dims[1]):
                        for i in range(self._n_voxels_dims[0]):
                            # Set element connectivities
                            connectivities[str(elem)] = [nodes_grid[i, j, k],
                                                         nodes_grid[i, j, k + 1],
                                                         nodes_grid[i + 1, j, k + 1],
                                                         nodes_grid[i + 1, j, k],
                                                         nodes_grid[i, j + 1, k],
                                                         nodes_grid[i, j + 1, k + 1],
                                                         nodes_grid[i + 1, j + 1, k + 1],
                                                         nodes_grid[i + 1, j + 1, k]]
                            # Set element material phase
                            element_phases[str(elem)] = self._regular_grid[i, j, k]
                            # Increment element counter
                            elem = elem + 1
            elif self._fe_order == 'quadratic':
                # Loop over elements
                for k in range(self._n_voxels_dims[2]):
                    for j in range(self._n_voxels_dims[1]):
                        for i in range(self._n_voxels_dims[0]):
                            # Set element connectivities
                            connectivities[str(elem)] = \
                                [nodes_grid[2*i, 2*j, 2*k],
                                 nodes_grid[2*i, 2*j, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j, 2*k],
                                 nodes_grid[2*i, 2*j + 2, 2*k],
                                 nodes_grid[2*i, 2*j + 2, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j + 2, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j + 2, 2*k],
                                 nodes_grid[2*i, 2*j, 2*k + 1],
                                 nodes_grid[2*i + 1, 2*j, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j, 2*k + 1],
                                 nodes_grid[2*i + 1, 2*j, 2*k],
                                 nodes_grid[2*i, 2*j + 1, 2*k],
                                 nodes_grid[2*i, 2*j + 1, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j + 1, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j + 1, 2*k],
                                 nodes_grid[2*i, 2*j + 2, 2*k + 1],
                                 nodes_grid[2*i + 1, 2*j + 2, 2*k + 2],
                                 nodes_grid[2*i + 2, 2*j + 2, 2*k + 1],
                                 nodes_grid[2*i + 1, 2*j + 2, 2*k]]
                            # Set element material phase
                            element_phases[str(elem)] = self._regular_grid[i, j, k]
                            # Increment element counter
                            elem = elem + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return coords, connectivities, element_phases
    # --------------------------------------------------------------------------------------
    def _run_links(self, file_path):
        '''Run Links code to solve microscale equilibrium problem.

        Parameters
        ----------
        file_path : str
            Links input data file path.
        '''
        # Call Links to solve microscale equilibrium problem
        subprocess.run([self._links_bin_path, file_path], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Links '.screen' file path
        screen_file_name = ntpath.splitext(ntpath.basename(file_path))[0]
        screen_file_path = ntpath.dirname(file_path) + '/' + screen_file_name + '/' + \
            screen_file_name + '.screen'
        # Check if the microscale equilibrium problem was successfully solved
        if not os.path.isfile(screen_file_path):
            raise RuntimeError('Links \'.screen\' file has not been found.')
        else:
            is_solved = False
            # Open '.screen' file
            screen_file = open(screen_file_path, 'r')
            screen_file.seek(0)
            # Look for succesful completion message
            line_number = 0
            for line in screen_file:
                line_number = line_number + 1
                if 'Program L I N K S successfully completed.' in line:
                    is_solved = True
                    break
            # Raise error if microscale equilibrium problem was not successfully solved
            if not is_solved:
                raise RuntimeError('Links could not successfully solve microscale '
                                   'equilibrium problem. Check \'.screen\' file for more '
                                   'details.')
    # --------------------------------------------------------------------------------------
    # Get the elementwise average strain tensor components
    def _get_strain_vox(self, links_file_path):
        '''Get local strain response from Links output file.

        Returns
        -------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str). Infinitesimal strain
            tensor (infinitesimal strains) or material logarithmic strain tensor (finite
            strains).
        '''
        # Initialize strain tensor
        strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                      for comp in self._comp_order_sym}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elementwise average output file path and check file existence
        elagv_file_name = ntpath.splitext(ntpath.basename(links_file_path))[0]
        elagv_file_path = ntpath.dirname(links_file_path) + '/' + elagv_file_name + '/' + \
            elagv_file_name + '.elavg'
        if not os.path.isfile(elagv_file_path):
            raise RuntimeError('Links elementwise average output file (\'.elagv\') has '
                               'not been found.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load elementwise average strain tensor components
        elagv_array = np.genfromtxt(elagv_file_path, autostrip=True)
        # Get Links strain components order
        links_comp_order_sym, _ = get_links_comp_order(self._n_dim)
        # Loop over Links strain components
        for i in range(len(links_comp_order_sym)):
            # Get Links strain component
            links_comp = links_comp_order_sym[i]
            # Set Links Voigt notation factor
            if links_comp[0] != links_comp[1]:
                voigt_factor = 2.0
            else:
                voigt_factor = 1.0
            # Store elementwise average strain component
            strain_vox[links_comp] = \
                (1.0/voigt_factor)*elagv_array[i, :].reshape(self._n_voxels_dims, order='F')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return strain_vox
