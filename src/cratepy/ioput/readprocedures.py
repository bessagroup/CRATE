"""Input data file reading and checking procedures.

This module includes a set of general and specific functions to read and
process data from the input data file. These functions also perform some checks
on the input data to avoid downstream execution errors.

Functions
---------
searchkeywordline
    Search mandatory keyword in data file and get corresponding line number.
searchoptkeywordline
    Search optional keyword in data file and get corresponding line number.
readtypeAkeyword
    Read keyword of specification type A.
readtypeBkeyword
    Read keyword of specification type B.
read_material_properties
    Read material phases data and properties.
read_macroscale_loading
    Read macroscale loading constraints.
read_mac_load_increm
    Read macroscale loading incrementation.
decode_increm_spec
    Decode macroscale loading increment specification.
read_phase_clustering
    Read (base) number of clusters associated with each material phase.
read_clustering_scheme
    Read material phase's prescribed clustering scheme.
check_clustering_scheme
    Check material phase's prescribed clustering scheme.
read_adaptivity_frequency
    Read clustering adaptivity frequency.
read_rewind_state_parameters
    Read the solution rewind state criterion parameters.
read_rewinding_criterion_parameters
    Read the solution rewinding criterion parameters.
read_discretization_file_path
    Read spatial discretization file path.
read_rve_dimensions
    Read RVE dimensions (size length along each spatial dimension).
read_self_consistent_scheme
    Read self-consistent scheme and associated parameters.
read_vtk_options
    Read VTK output options.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
import linecache
import re
# Third-party
import numpy as np
# Local
import ioput.info as info
import ioput.ioutilities as ioutil
from clustering.crve import CRVE
from clustering.clusteringphase import SCRMP
from clustering.adaptivity.crve_adaptivity import AdaptivityManager
from online.loading.macloadincrem import RewindManager
from material.materialmodeling import get_available_material_models
from material.models.elastic import Elastic
from material.models.von_mises import VonMises
from material.models.stvenant_kirchhoff import StVenantKirchhoff
from online.crom.asca import ElasticReferenceMaterial
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                                               Search keywords
# =============================================================================
def searchkeywordline(file, keyword):
    """Search mandatory keyword in data file and get corresponding line number.

    Parameters
    ----------
    file : file
        Data file.
    keyword : str
        Keyword.

    Returns
    -------
    line_number : int
        Number of data file line where the keyword is first found.
    """
    file.seek(0)
    line_number = 0
    for line in file:
        line_number = line_number + 1
        if keyword in line.split() and line.strip()[0] != '#':
            return line_number
    # Keyword not found
    summary = 'Missing keyword'
    description = 'The keyword - {} - has not been found in the input ' \
        + 'data file.'
    info.displayinfo('4', summary, description, keyword)
# =============================================================================
def searchoptkeywordline(file, keyword):
    """Search optional keyword in data file and get corresponding line number.

    Parameters
    ----------
    file : file
        Data file.
    keyword : str
        Keyword.

    Returns
    -------
    is_found : bool
        `True` if keyword is found in data file, `False` otherwise.
    line_number : int
        Number of data file line where the keyword is first found. Set to 0 by
        default if keyword is not found.
    """
    is_found = False
    file.seek(0)
    line_number = 0
    for line in file:
        line_number = line_number + 1
        if keyword in line.split() and line.strip()[0] != '#':
            is_found = True
            return is_found, line_number
    return is_found, line_number
#
#                                                           Parameter formatter
# =============================================================================
def get_formatted_parameter(parameter, x, etype=None):
    """Get string parameter converted to appropriate type.

    Parameters
    ----------
    parameter : str
        Parameter name.
    x : str
        Parameter specification.
    etype : {int, float, str, bool}, default=None
        Parameter expected type.

    Returns
    -------
    y : {'int', 'float', 'str', 'bool'}
        Parameter value.
    """
    # Get parameter specification type and associated value
    try:
        a = float(x)
        b = int(a)
        if a == b and len(str(x)) == len(str(b)):
            stype_name = 'int'
            y = int(x)
        else:
            stype_name = 'float'
            y = float(x)
    except (TypeError, ValueError):
        if x.lower() == 'on':
            stype_name = 'bool'
            y = True
        elif x.lower() == 'off':
            stype_name = 'bool'
            y = False
        else:
            stype_name = 'str'
            y = x
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if specification type agrees with expected type
    if etype is not None:
        if stype_name != etype.__name__:
            raise TypeError('The parameter \'' + str(parameter)
                            + '\' hasn\'t been properly specified: expected '
                            + etype.__name__ + ' but found ' + stype_name
                            + '.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return y
#
#                                                              General keywords
# =============================================================================
def readtypeAkeyword(file, file_path, keyword, max_val):
    """Read keyword of specification type A.

    The keyword specification of type A is associated with a positive
    integer-valued keyword and has the following input data file syntax:

    .. code-block:: text

       < keyword > < int >

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword : str
        Keyword.
    max_val : int
        Maximum value of keyword positive integer value.

    Returns
    -------
    keyword_value : int
        Keyword value.
    """
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number).split()
    if len(line) == 1:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    elif not ioutil.checkposint(line[1]):
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    elif isinstance(max_val, int) or isinstance(max_val, np.integer):
        if int(line[1]) > max_val:
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.'
            info.displayinfo('4', summary, description, keyword)
    return int(line[1])
# =============================================================================
def readtypeBkeyword(file, file_path, keyword):
    """Read keyword of specification type B.

    The keyword specification of type B is associated with a positive
    float-valued keyword and has the following input data file syntax:

    .. code-block:: text

       < keyword >
       < float >

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword : str
        Keyword.

    Returns
    -------
    keyword_value : float
        Keyword value.
    """
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number+1).split()
    if line == '':
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    elif len(line) != 1:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    elif not ioutil.checknumber(line[0]) or float(line[0]) <= 0:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    return float(line[0])
#
#                                                             Specific keywords
# =============================================================================
def read_material_properties(file, file_path, keyword):
    """Read material phases data and properties.

    The specification of the data associated with the material phases has the
    following input data file syntax:

    .. code-block:: text

        Material_Phases < n_material_phases >
        < phase_id > < model_name > < n_prop_copt > [ < model_source > ]
        < property_1_name > < value >
        < property_2_name > < value >
        < constitutive_option_1_name > < option > [ < n_coproperties > ]
            < coproperty_1_name > < value >
            < coproperty_2_name > < value >
        < phase_id > < model_name > < n_prop_copt > [ < model_source > ]
        < property1_name > < value >
        < constitutive_option_1_name > < option > [ < n_coproperties > ]
            < coproperty_1_name > < value >
            < coproperty_2_name > < value >
        < property2_name > < value >
        ...

    where `n_material_phases` (int) is the number of material phases,
    `phase_id` (int) is the material identifier , `model_name` (str) is the
    constitutive model name, `n_prop_copt` (int) is the total number of
    constitutive properties and options, `model_source` (int, optional) is the
    constitutive model source, `property_X_name` (str) is the constitutive
    property name, `constitutive_option_X_name` is the constitutive option
    name, `n_coproperties` is the number of properties associated with the
    constitutive option, and `coproperty_X_name` is the constitutive option
    property name.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword : str
        Keyword.

    Returns
    -------
    n_material_phases : int
        Number of material phases.
    material_phases_data : dict
        Material phase data (item, dict) associated with each material phase
        (key, str).
    material_phases_properties : dict
        Constitutive model material properties (item, dict) associated with
        each material phase (key, str).
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Search keyword
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number).split()
    if len(line) == 1:
        summary = 'Missing number of material phases'
        description = 'The keyword - {} - is not properly defined in ' \
            + 'the input data file.' + '\n' \
            + indent + 'Missing number of material phases.'
        info.displayinfo('4', summary, description, keyword)
    elif not ioutil.checkposint(line[1]):
        summary = 'Invalid number of material phases'
        description = 'The keyword - {} - is not properly defined in ' \
            + 'the input data file.' + '\n' \
            + indent + 'Invalid number of material phases.'
        info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of material phases
    n_material_phases = int(line[1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material phases properties and constitutive models
    # dictionaries
    material_phases_properties = {}
    material_phases_data = {}
    # Loop over material phases
    line_number = keyword_line_number + 1
    for i in range(n_material_phases):
        # Read material phase header
        phase_header = linecache.getline(file_path, line_number).split()
        if phase_header[0] == '':
            summary = 'Missing material phase header'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + indent + 'Missing specification of a material phase header.'
            info.displayinfo('4', summary, description, keyword)
        elif len(phase_header) not in [3, 4]:
            summary = 'Missing material phase header'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + indent + 'Missing specification of a material phase header.'
            info.displayinfo('4', summary, description, keyword)
        elif not ioutil.checkposint(phase_header[0]):
            summary = 'Invalid material phase header'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + indent + 'Invalid specification of a material phase header.'
            info.displayinfo('4', summary, description, keyword)
        elif phase_header[0] in material_phases_properties.keys():
            summary = 'Duplicated material phase header'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + indent + 'Duplicated specification of a material phase ' \
                + 'header.'
            info.displayinfo('4', summary, description, keyword)
        # Set material phase
        mat_phase = str(phase_header[0])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        material_phases_data[mat_phase] = {}
        # Get material phase constitutive model source identifier
        if len(phase_header) == 3:
            # If the material phase constitutive model source has not been
            # specified, then assume CRATE by default
            model_source_id = 1
        elif len(phase_header) == 4:
            # Set constitutive model source
            if not ioutil.checkposint(phase_header[3]):
                summary = 'Invalid constitutive model source'
                description = 'The keyword - {} - is not properly defined in '\
                    + 'the input data file.' + '\n' \
                    + indent + 'Invalid constitutive model source of ' \
                    + 'material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            else:
                model_source_id = int(phase_header[3])
        # Set material phase constitutive model source
        if model_source_id == 1:
            model_source = 'crate'
        else:
            summary = 'Unknown constitutive model source'
            description = 'The keyword - {} - is not properly defined in '\
                + 'the input data file.' + '\n' \
                + indent + 'Unknown constitutive model source of ' \
                + 'material phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)
        # Assemble material phase constitutive model source
        material_phases_data[mat_phase]['source'] = model_source
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get available material constitutive models
        available_mat_models = get_available_material_models(model_source)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material phase constitutive model keyword
        if phase_header[1] not in available_mat_models:
            summary = 'Unknown material constitutive model'
            description = 'The keyword - {} - is not properly defined in '\
                + 'the input data file.' + '\n' \
                + indent + 'Unknown material constitutive model of ' \
                + 'material phase {}.'
            info.displayinfo('4', summary, description, keyword,
                             phase_header[1])
        else:
            model_keyword = phase_header[1]
        material_phases_data[mat_phase]['keyword'] = model_keyword
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material constitutive model constitutive options and material
        # properties
        if model_source == 'crate':
            if model_keyword == 'elastic':
                req_properties, req_constitutive_options = \
                    Elastic.get_required_properties()
            elif model_keyword == 'von_mises':
                req_properties, req_constitutive_options = \
                    VonMises.get_required_properties()
            elif model_keyword == 'stvenant_kirchhoff':
                req_properties, req_constitutive_options = \
                    StVenantKirchhoff.get_required_properties()
        # Set number of constitutive options and material properties
        n_prop_copt = len(req_properties) \
            + len(req_constitutive_options.keys())
        if not ioutil.checkposint(phase_header[2]):
            summary = 'Invalid number of material properties'
            description = 'The keyword - {} - is not properly defined in '\
                + 'the input data file.' + '\n' \
                + indent + 'Invalid number of material properties and ' \
                + 'constitutive options of material phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)
        elif int(phase_header[2]) != n_prop_copt:
            summary = 'Wrong number of material properties'
            description = 'The keyword - {} - is not properly defined in '\
                + 'the input data file.' + '\n' \
                + indent + 'Wrong number of material properties and ' \
                + 'constitutive options of material phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update line number
        line_number = line_number + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set constitutive options and all material properties
        material_phases_properties[mat_phase] = {}
        # Loop over constitutive options and material properties
        for j in range(n_prop_copt):
            property_header_line = line_number
            property_line = \
                linecache.getline(file_path, property_header_line).split()
            if property_line[0] == '':
                summary = 'Invalid material property or constitutive option'
                description = 'The keyword - {} - is not properly defined in '\
                    + 'the input data file.' + '\n' \
                    + indent + 'Invalid material property or constitutive ' \
                    + 'option of material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            elif not ioutil.checkvalidname(property_line[0]):
                summary = 'Invalid material property or constitutive option'
                description = 'The keyword - {} - is not properly defined in '\
                    + 'the input data file.' + '\n' \
                    + indent + 'Invalid material property or constitutive ' \
                    + 'option of material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            elif property_line[0] not in req_properties and \
                    property_line[0] not in req_constitutive_options.keys():
                summary = 'Invalid material property or constitutive option'
                description = 'The keyword - {} - is not properly defined in '\
                    + 'the input data file.' + '\n' \
                    + indent + 'Invalid material property or constitutive ' \
                    + 'option of material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            elif property_line[0] in \
                    material_phases_properties[mat_phase].keys():
                summary = 'Duplicated material property or constitutive option'
                description = 'The keyword - {} - is not properly defined in '\
                    + 'the input data file.' + '\n' \
                    + indent + 'Duplicated material property or constitutive '\
                    + 'option of material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Read material property or constitutive option (and associated
            # material properties)
            if str(property_line[0]) in req_constitutive_options.keys():
                # Check if constitutive option specification is available
                if str(property_line[1]) not in \
                        req_constitutive_options[str(property_line[0])]:
                    summary = 'Unknown constitutive option'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Unknown constitutive option of material '\
                        + 'phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     mat_phase)
                else:
                    constitutive_option = str(property_line[1])
                # Assemble constitutive option specification
                material_phases_properties[mat_phase][
                    str(property_line[0])] = constitutive_option
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Read constitutive option number of associated properties
                if len(property_line) == 3:
                    if not ioutil.checkposint(property_line[2]):
                        summary = 'Invalid number of properties of ' \
                            + 'constitutive option'
                        description = 'The keyword - {} - is not properly ' \
                            + 'defined in the input data file.' + '\n' \
                            + indent + 'Invalid number of constitutive ' \
                            + 'options of material phase {}.'
                        info.displayinfo('4', summary, description, keyword,
                                         mat_phase)
                    else:
                        n_coproperties = int(property_line[2])
                elif len(property_line) == 2:
                    n_coproperties = 1
                else:
                    summary = 'Invalid constitutive option'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Invalid constitutive option of material ' \
                        + 'phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     mat_phase)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if str(property_line[0]) == 'isotropic_hardening' and \
                        constitutive_option == 'piecewise_linear':
                    # Initialize hardening points
                    hardening_points = np.zeros((n_coproperties, 2))
                    # Read hardening points
                    for k in range(n_coproperties):
                        # Read hardening point line
                        hardening_point_line = linecache.getline(
                            file_path, property_header_line + 1 + k).split()
                        if hardening_point_line[0] == '':
                            summary = 'Invalid hardening point'
                            description = 'The specification of a strain ' \
                                + 'hardening point of material phase {} in ' \
                                + 'the input data' + '\n' \
                                + indent + 'file is invalid.'
                            info.displayinfo('4', summary, description,
                                             mat_phase)
                        elif len(hardening_point_line) != 3:
                            summary = 'Invalid hardening point'
                            description = 'The specification of a strain ' \
                                + 'hardening point of material phase {} in ' \
                                + 'the input data' + '\n' \
                                + indent + 'file is invalid.'
                            info.displayinfo('4', summary, description,
                                             mat_phase)
                        elif not ioutil.checknumber(hardening_point_line[1]) \
                                or not ioutil.checknumber(
                                    hardening_point_line[2]):
                            summary = 'Invalid hardening point'
                            description = 'The specification of a strain ' \
                                + 'hardening point of material phase {} in ' \
                                + 'the input data' + '\n' \
                                + indent + 'file is invalid.'
                            info.displayinfo('4', summary, description,
                                             mat_phase)
                        hardening_points[k, 0] = float(hardening_point_line[1])
                        hardening_points[k, 1] = float(hardening_point_line[2])
                    # Assemble constitutive parameter associated property
                    material_phases_properties[mat_phase][
                        'hardening_points'] = hardening_points
                else:
                    # Read constitutive option associated properties
                    for k in range(n_coproperties):
                        # Read constitutive option line
                        coproperty_line = linecache.getline(
                            file_path, property_header_line + 1 + k).split()
                        if len(coproperty_line) < 2:
                            summary = 'Invalid constitutive option'
                            description = 'The keyword - {} - is not ' \
                                + 'properly defined in the input data file.' \
                                + '\n' \
                                + indent + 'Invalid constitutive option of ' \
                                + 'material phase {}.'
                            info.displayinfo('4', summary, description,
                                             keyword, mat_phase)
                        # Get property name
                        prop_name = str(coproperty_line[0])
                        # Get property value
                        if len(coproperty_line) == 2:
                            # Single value
                            prop_value = get_formatted_parameter(
                                coproperty_line[0], coproperty_line[1])
                        else:
                            # Multiple values stored as tuple
                            prop_value = tuple(
                                (get_formatted_parameter(coproperty_line[0],
                                                         coproperty_line[x])
                                 for x in range(1, len(coproperty_line))))
                        # Assemble constitutive parameter associated property
                        material_phases_properties[mat_phase][prop_name] = \
                            prop_value
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update line number
                line_number = line_number + n_coproperties
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                if len(property_line) != 2:
                    summary = 'Invalid material property'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Invalid material property of material ' \
                        + 'phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     mat_phase)
                elif not ioutil.checknumber(property_line[1]):
                    summary = 'Invalid material property'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Invalid material property of material ' \
                        + 'phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     mat_phase)
                prop_name = str(property_line[0])
                prop_value = float(property_line[1])
                material_phases_properties[mat_phase][prop_name] = prop_value
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update line number
            line_number = line_number + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return n_material_phases, material_phases_data, material_phases_properties
# =============================================================================
def read_macroscale_loading(file, file_path, mac_load_type, strain_formulation,
                            n_dim, comp_order_nsym):
    """Read macroscale loading constraints.

    The storage of the macroscale strain/stress tensors is performed according
    to the problem type nonsymmetric component order.

    The specification of the data associated with the macroscale loading
    constraints has the following input data file syntax:

    **2D Problem:**

    *Macroscale strain constraint:*

    .. code-block:: text

       Macroscale_Strain [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...

    where `n_load_subpaths` is the number of loading subpaths (defaults to 1),
    each associated with a given column. Irrespective of the name given to each
    component, it is assumed that all the components of the macroscale strain
    tensor (assumed nonsymmetric) are specified in columnwise order.

    *Macroscale stress constraint:*

    .. code-block:: text

       Macroscale_Stress [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...

    where `n_load_subpaths` is the number of loading subpaths (defaults to 1),
    each associated with a given column. Irrespective of the name given to each
    component, it is assumed that all the components of the macroscale stress
    tensor (assumed nonsymmetric) are specified in columnwise order.

    *Macroscale strain and stress constraint:*

    .. code-block:: text

       Macroscale_Strain [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...

       Macroscale_Stress < n_load_subpaths >
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...

       Mixed_Prescription_Index
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...

    where `n_load_subpaths` is the number of loading subpaths (defaults to 1),
    each associated with a given column. The corresponding nature of each
    strain (0) or stress (1) component is specified accordingly under the
    `Mixed_Prescription_Index` keyword. Irrespective of the name given to each
    component, it is assumed that all the components of the macroscale strain
    and stress tensors (assumed nonsymmetric) are specified in columnwise
    order. Components not enforced through the mixed loading prescription are
    ignored.

    ----

    **3D Problem:**

    *Macroscale strain constraint:*

    .. code-block:: text

       Macroscale_Strain [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_31 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...
       < component_name_32 > < float > < float >  ...
       < component_name_13 > < float > < float >  ...
       < component_name_23 > < float > < float >  ...
       < component_name_33 > < float > < float >  ...

    where `n_load_subpaths` is the number of loading subpaths (defaults to 1),
    each associated with a given column. Irrespective of the name given to each
    component, it is assumed that all the components of the macroscale strain
    tensor (assumed nonsymmetric) are specified in columnwise order.

    *Macroscale stress constraint:*

    .. code-block:: text

       Macroscale_Stress [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_31 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...
       < component_name_32 > < float > < float >  ...
       < component_name_13 > < float > < float >  ...
       < component_name_23 > < float > < float >  ...
       < component_name_33 > < float > < float >  ...

    where `n_load_subpaths` is the number of loading subpaths (defaults to 1),
    each associated with a given column. Irrespective of the name given to each
    component, it is assumed that all the components of the macroscale stress
    tensor (assumed nonsymmetric) are specified in columnwise order.

    *Macroscale strain and stress constraint:*

    .. code-block:: text

       Macroscale_Strain [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_31 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...
       < component_name_32 > < float > < float >  ...
       < component_name_13 > < float > < float >  ...
       < component_name_23 > < float > < float >  ...
       < component_name_33 > < float > < float >  ...

       Macroscale_Stress [< n_load_subpaths >]
       < component_name_11 > < float > < float >  ...
       < component_name_21 > < float > < float >  ...
       < component_name_31 > < float > < float >  ...
       < component_name_12 > < float > < float >  ...
       < component_name_22 > < float > < float >  ...
       < component_name_32 > < float > < float >  ...
       < component_name_13 > < float > < float >  ...
       < component_name_23 > < float > < float >  ...
       < component_name_33 > < float > < float >  ...

       Mixed_Prescription_Index
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...
       < 0 or 1 > < 0 or 1 >  ...

    where `n_load_subpaths` is the number of loading subpaths (defaults to 1),
    each associated with a given column. The corresponding nature of each
    strain (0) or stress (1) component is specified accordingly under the
    `Mixed_Prescription_Index` keyword. Irrespective of the name given to each
    component, it is assumed that all the components of the macroscale strain
    and stress tensors (assumed nonsymmetric) are specified in columnwise
    order. Components not enforced through the mixed loading prescription are
    ignored.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    mac_load_type : {1, 2, 3}
        Macroscale loading type:

        * 1 : Macroscale strain constraint
        * 2 : Macroscale stress constraint
        * 3 : Macroscale strain and stress constraint
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.

    Returns
    -------
    mac_load : dict
        For each loading nature type (key, {'strain', 'stress'}), stores
        the loading constraints for each loading subpath in a
        numpy.ndarray (2d), where the i-th row is associated with the i-th
        strain/stress component and the j-th column is associated with the
        j-th loading subpath.
    mac_load_presctype : numpy.ndarray (2d)
        Loading nature type ({'strain', 'stress'}) associated with each
        loading constraint (ndarray of shape (n_comps, n_load_subpaths)),
        where the i-th row is associated with the i-th strain/stress
        component and the j-th column is associated with the j-th loading
        subpath.
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale loading keywords according to loading type
    if mac_load_type == 1:
        loading_keywords = {'Macroscale_Strain': 'strain'}
    elif mac_load_type == 2:
        loading_keywords = {'Macroscale_Stress': 'stress'}
    elif mac_load_type == 3:
        loading_keywords = {'Macroscale_Strain': 'strain',
                            'Macroscale_Stress': 'stress'}
        presc_keyword = 'Mixed_Prescription_Index'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize macroscale loading dictionary
    mac_load = {key: None for key in ['strain', 'stress']}
    # Initialize number of macroscale loading subpaths dictionary
    n_load_subpaths = {key: 0 for key in ['strain', 'stress']}
    # Loop over macroscale loading keywords
    for load_key in loading_keywords.keys():
        # Get load nature type
        ltype = loading_keywords[load_key]
        # Get macroscale loading keyword line number
        load_keyword_line_number = searchkeywordline(file, load_key)
        # Check number of loading subpaths
        keyword_line = \
            linecache.getline(file_path, load_keyword_line_number).split()
        if len(keyword_line) > 2:
            summary = 'Invalid number of loading subpaths'
            description = 'The specification of the number of macroscale ' \
                + 'loading subpaths in the input data file ' + '\n' \
                + indent + 'is invalid.'
            info.displayinfo('4', summary, description)
        elif len(keyword_line) == 2:
            if ioutil.checkposint(keyword_line[1]):
                n_load_subpaths[ltype] = int(keyword_line[1])
            else:
                summary = 'Invalid number of loading subpaths'
                description = 'The specification of the number of macroscale '\
                    + 'loading subpaths in the input data file ' + '\n' \
                    + indent + 'is invalid.'
                info.displayinfo('4', summary, description)
        else:
            n_load_subpaths[ltype] = 1
        # Initialize macroscale loading array
        mac_load[ltype] = np.full((n_dim**2, 1 + n_load_subpaths[ltype]), 0.0,
                                  dtype=object)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale loading
    for load_key in loading_keywords:
        # Get load nature type
        ltype = loading_keywords[load_key]
        load_keyword_line_number = searchkeywordline(file, load_key)
        # Loop over macroscale loading components
        for i_comp in range(n_dim**2):
            component_line = linecache.getline(
                file_path, load_keyword_line_number + i_comp + 1).split()
            if not component_line:
                summary = 'Invalid keyword specification'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + 'Check {} th component.'
                info.displayinfo('4', summary, description, load_key,
                                 i_comp + 1)
            elif len(component_line) != 1 + n_load_subpaths[ltype]:
                summary = 'Invalid keyword specification'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + 'Check {} th component.'
                info.displayinfo('4', summary, description, load_key,
                                 i_comp + 1)
            elif not ioutil.checkvalidname(component_line[0]):
                summary = 'Invalid keyword specification'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + 'Check {} th component.'
                info.displayinfo('4', summary, description, load_key,
                                 i_comp + 1)
            # Set component name
            mac_load[ltype][i_comp, 0] = component_line[0]
            # Set component values for each loading subpath
            for j in range(n_load_subpaths[ltype]):
                presc_val = component_line[1 + j]
                if not ioutil.checknumber(presc_val):
                    summary = 'Invalid keyword specification'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + 'Check {} th component.'
                    info.displayinfo('4', summary, description, load_key,
                                     i_comp + 1)
                else:
                    mac_load[ltype][i_comp, 1 + j] = float(presc_val)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale prescription nature indexes
    if mac_load_type == 1:
        ltype = loading_keywords['Macroscale_Strain']
        mac_load_presctype = np.full((n_dim**2, n_load_subpaths[ltype]),
                                     'strain', dtype=object)
    elif mac_load_type == 2:
        ltype = loading_keywords['Macroscale_Stress']
        mac_load_presctype = np.full((n_dim**2, n_load_subpaths[ltype]),
                                     'stress', dtype=object)
    elif mac_load_type == 3:
        mac_load_presctype = np.full((n_dim**2, max(n_load_subpaths.values())),
                                     'ND', dtype=object)
        presc_keyword = 'Mixed_Prescription_Index'
        presc_keyword_line_number = searchkeywordline(file, presc_keyword)
        # Loop over macroscale loading components
        for i_comp in range(n_dim**2):
            component_line = linecache.getline(
                file_path, presc_keyword_line_number + i_comp + 1).split()
            if not component_line:
                summary = 'Invalid keyword specification'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + 'Check {} th component.'
                info.displayinfo('4', summary, description, load_key,
                                 i_comp + 1)
            # Set prescription nature indexes for each loading subpath
            for j in range(max(n_load_subpaths.values())):
                presc_val = int(component_line[j])
                if presc_val not in [0, 1]:
                    summary = 'Invalid keyword specification'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + 'Check {} th component.'
                    info.displayinfo('4', summary, description, load_key,
                                     i_comp + 1)
                else:
                    ltype = 'strain' if presc_val == 0 else 'stress'
                    if j >= n_load_subpaths[ltype]:
                        summary = 'Invalid keyword specification'
                        description = 'The keyword - {} - is not properly ' \
                            + 'defined in the input data file.' + '\n' \
                            + 'Check {} th component.'
                        info.displayinfo('4', summary, description, load_key,
                                         i_comp + 1)
                    else:
                        mac_load_presctype[i_comp, j] = ltype
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check small strain formulation symmetry
    if strain_formulation == 'infinitesimal':
        # Set symmetric indexes (columnwise)
        if n_dim**2 == 4:
            symmetric_indexes = np.array([[2], [1]])
        elif n_dim**2 == 9:
            symmetric_indexes = np.array([[3, 6, 7], [1, 2, 5]])
        # Loop over symmetric indexes
        for i in range(symmetric_indexes.shape[1]):
            # Loop over loading subpaths
            for j in range(max(n_load_subpaths.values())):
                # Get load nature type
                ltype = mac_load_presctype[symmetric_indexes[0, i], j]
                if mac_load_type == 3 \
                        and mac_load_presctype[symmetric_indexes[0, i], j] \
                        != mac_load_presctype[symmetric_indexes[1, i], j]:
                    summary = 'Symmetric components prescribed with ' \
                        + 'different nature under infinitesimal strains'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Symmetric components must have the same ' \
                        + 'nature (strain or stress). Check {} th component.'
                    info.displayinfo('4', summary, description,
                                     'Mixed_Prescription_Index', i_comp + 1)
                # Check symmetry
                isEqual = np.allclose(
                    mac_load[ltype][symmetric_indexes[0, i], j + 1],
                    mac_load[ltype][symmetric_indexes[1, i], j + 1],
                    atol=1e-10)
                if not isEqual:
                    summary = 'Nonsymmetric strain or stress components ' \
                        + 'prescribed under infinitesimal strains'
                    description = 'A nonsymmetric {} tensor is prescribed ' \
                        + 'in the input data file under infinitesimal strains.'
                    info.displayinfo('4', summary, description, ltype)
                    # Adopt symmetric component with the lowest first index
                    mac_load[ltype][symmetric_indexes[1, i], j + 1] = \
                        mac_load[ltype][symmetric_indexes[0, i], j + 1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort macroscale strain and stress tensors according to the defined
    # problem nonsymmetric component order
    if n_dim == 2:
        aux = {'11': 0, '21': 1, '12': 2, '22': 3}
    else:
        aux = {'11': 0, '21': 1, '31': 2, '12': 3, '22': 4, '32': 5, '13': 6,
               '23': 7, '33': 8}
    mac_load_copy = copy.deepcopy(mac_load)
    mac_load_presctype_copy = copy.deepcopy(mac_load_presctype)
    for i in range(n_dim**2):
        if mac_load_type == 1:
            mac_load['strain'][i, :] = \
                mac_load_copy['strain'][aux[comp_order_nsym[i]], :]
        elif mac_load_type == 2:
            mac_load['stress'][i, :] = \
                mac_load_copy['stress'][aux[comp_order_nsym[i]], :]
        elif mac_load_type == 3:
            mac_load['strain'][i, :] = \
                mac_load_copy['strain'][aux[comp_order_nsym[i]], :]
            mac_load['stress'][i, :] = \
                mac_load_copy['stress'][aux[comp_order_nsym[i]], :]
            mac_load_presctype[i, :] = \
                mac_load_presctype_copy[aux[comp_order_nsym[i]], :]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mac_load, mac_load_presctype
# =============================================================================
def read_mac_load_increm(file, file_path, keyword, n_load_subpaths):
    """Read macroscale loading incrementation.

    The specification of the data associated with the macroscale loading
    constraints has the following input data file syntax:

    *Option 1: Number_of_Load_Increments*

    This option entails a given number of equal-magnitude loading increments
    applied to each loading subpath.

    .. code-block:: text

       Number_of_Load_Increments < int >


    *Option 2: Increment_List*

    This option provides a general loading incrementation scheme.

    .. code-block:: text

       Increment_List
       [n_rep:] inc_load_fact[_inc_time] | [n_rep:] inc_load_fact[_inc_time] ..
       [n_rep:] inc_load_fact[_inc_time] | [n_rep:] inc_load_fact[_inc_time] ..
       [n_rep:] inc_load_fact[_inc_time] | [n_rep:] inc_load_fact[_inc_time] ..

    where `n_rep` (int) is the number of increment repetitions (optional,
    defaults to 1), `inc_load_fact` is the incremental load factor, and
    `inc_time` is the incremental time (optional, defaults to loading time
    factor times the absolute value of the incremental load factor).
    The delimiter `|` separates different loading subpaths.

    ----

    The optional specification of the data associated with the macroscale
    loading time factor has the following input data file syntax:

    .. code-block:: text

       Loading_Time_Factor
       < float >

    Unless explicitly specified in any way, the loading time factor provides
    the time associated with the macroscale loading incrementation. The
    incremental time is obtained by multiplying the loading time factor by the
    incremental loading factor. Defaults to 1.0 if not specified.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword : str
        Keyword.
    n_load_subpaths : int
        Number of loading subpaths.

    Returns
    -------
    mac_load_increm : dict
        For each loading subpath id (key, str), stores a numpy.ndarray of shape
        (n_load_increments, 2) where each row is associated with a prescribed
        loading increment, and the columns 0 and 1 contain the corresponding
        incremental load factor and incremental time, respectively.
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize macroscale loading incrementation dictionary
    mac_load_increm = dict()
    # Set loading time factor
    keyword_time = 'Loading_Time_Factor'
    is_found, _ = searchoptkeywordline(file, keyword_time)
    if is_found:
        load_time_factor = readtypeBkeyword(file, file_path, keyword_time)
    else:
        load_time_factor = 1.0
    # Set macroscale loading incrementation
    if keyword == 'Number_of_Load_Increments':
        max_val = '~'
        n_load_increments = readtypeAkeyword(file, file_path, keyword, max_val)
        # Build macroscale loading incrementation dictionary
        for i in range(n_load_subpaths):
            # Set loading subpath default total load factor
            total_lfact = 1.0
            # Build macroscale loading subpath
            load_subpath = np.zeros((n_load_increments, 2))
            load_subpath[:, 0] = total_lfact/n_load_increments
            load_subpath[:, 1] = load_time_factor*load_subpath[:, 0]
            # Store macroscale loading subpath
            mac_load_increm[str(i)] = load_subpath
    elif keyword == 'Increment_List':
        # Find keyword line number
        keyword_line_number = searchkeywordline(file, keyword)
        # Initialize macroscale loading increment array
        increm_list = np.full((0, n_load_subpaths), '', dtype=object)
        # Read increment specification line
        line = linecache.getline(file_path, keyword_line_number + 1)
        increm_line = [x.strip() for x in line.split('|')]
        # At least one increment specification line must be provided for each
        # macroscale loading subpath
        is_empty_line = not bool(line.split())
        if is_empty_line or len(increm_line) != n_load_subpaths:
            summary = 'Invalid loading increment list'
            description = 'The keyword - {} - is not properly defined' \
                + ' in the input data file.' + '\n' \
                + indent + 'The first line of the increment list must ' \
                + 'contain only one macroscale loading increment' + '\n' + \
                + indent + 'specification for each macroscale loading subpath.'
            info.displayinfo('4', summary, description, keyword)
        i = 0
        # Build macroscale loading increment array
        while not is_empty_line:
            increm_list = np.append(increm_list,
                                    np.full((1, n_load_subpaths), '',
                                            dtype=object), axis=0)
            # Assemble macroscale increment specification line
            increm_list[i, 0:len(increm_line)] = increm_line
            i += 1
            # Read increment specification line
            line = linecache.getline(file_path, keyword_line_number + 1 + i)
            is_empty_line = not bool(line.split())
            increm_line = [x.strip() for x in line.split('|')]
        # Build macroscale loading incrementation dictionary
        for j in range(n_load_subpaths):
            # Initialize macroscale loading subpath
            load_subpath = np.zeros((0, 2))
            # Loop over increment specifications
            for i in range(increm_list.shape[0]):
                # Get increment specification
                spec = increm_list[i, j]
                # Decode increment specification
                if spec == '':
                    break
                else:
                    rep, inc_lfact, inc_time = \
                        decode_increm_spec(spec, load_time_factor)
                # Build macroscale loading subpath
                load_subpath = np.append(load_subpath,
                                         np.tile([inc_lfact, inc_time],
                                                 (rep, 1)), axis=0)
            # Store macroscale loading subpath
            mac_load_increm[str(j)] = load_subpath
    else:
        # Unknown macroscale loading keyword
        summary = 'Unknown loading incrementation keyword'
        description = 'A unknown macroscale loading incrementation keyword ' \
            + 'has been specified in the input data file.'
        info.displayinfo('4', summary, description)
    # Return
    return mac_load_increm
# =============================================================================
def decode_increm_spec(spec, load_time_factor):
    """Decode macroscale loading increment specification.

    A macroscale loading increment specification has the format
    `[n_rep:] inc_load_fact[_inc_time]` where `n_rep` (int) is the number of
    increment repetitions (optional, defaults to 1), `inc_load_fact` is the
    incremental load factor, and `inc_time` is the incremental time (optional,
    defaults to loading time factor times the absolute value of the incremental
    load factor).

    ----

    Parameters
    ----------
    spec : str
        Macroscale loading increment specification.
    load_time_factor : float
        Loading time factor.

    Returns
    -------
    n_rep : int
        Number of increment repetitions.
    inc_lfact : float
        Incremental load factor.
    inc_time : float
        Incremental time.
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Split specifications based on multiple delimiters
    code = re.split('[:_]', spec)
    # Check if the repetition and incremental time have been specified
    has_rep = ':' in re.findall('[:_]', spec)
    has_time = '_' in re.findall('[:_]', spec)
    if not code or len(code) > 3:
        summary = 'Invalid loading increment'
        description = 'A macroscale loading increment specification in ' \
            + 'the input data file is invalid.'
        info.displayinfo('4', summary, description)
    # Set macroscale loading increment parameters
    try:
        n_rep = int(code[0]) if has_rep else 1
        inc_lfact = float(code[int(has_rep)])
        inc_time = abs(float(code[-1])) \
            if has_time else load_time_factor*abs(inc_lfact)
    except Exception:
        summary = 'Invalid loading increment'
        description = 'A macroscale loading increment specification in ' \
            + 'the input data file is invalid.'
        info.displayinfo('4', summary, description)
    else:
        if any([x < 0 for x in [n_rep, inc_time]]):
            summary = 'Invalid loading increment optinal parameters'
            description = 'The number of repetitions or incremental time ' \
                + 'prescribed for a given macroscale loading' + '\n' \
                + indent + 'increment in the input data file is invalid.'
            info.displayinfo('4', summary, description)
    # Return
    return n_rep, inc_lfact, inc_time
# =============================================================================
def read_phase_clustering(file, file_path, keyword, n_material_phases,
                          material_properties):
    """Read (base) number of clusters associated with each material phase.

    The specification of the data associated with the material phases (base)
    number of clusters has the following input data file syntax:

    .. code-block:: text

       Number_of_Clusters
       < phase_id > < n_clusters >
       < phase_id > < n_clusters >
       ...

    where `phase_id` (int) is the material identifier and `n_clusters` is the
    corresponding (base) number of clusters.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword : str
        Keyword.
    n_material_phases : int
        Number of material phases.
    material_properties : dict
        Constitutive model material properties (key, str) values
        (item, {int, float, bool}).

    Returns
    -------
    phase_n_clusters : dict
        Number of clusters (item, int) associated with each material phase
        (key, str).
    """
    phase_n_clusters = dict()
    line_number = searchkeywordline(file, keyword) + 1
    for iphase in range(n_material_phases):
        line = linecache.getline(file_path, line_number + iphase).split()
        if line[0] == '':
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + 'Check {}th material phase.'
            info.displayinfo('4', summary, description, keyword, iphase + 1)
        elif len(line) != 2:
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + 'Check {}th material phase.'
            info.displayinfo('4', summary, description, keyword, iphase + 1)
        elif not ioutil.checkposint(line[0]) \
                or not ioutil.checkposint(line[1]):
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + 'Check {}th material phase.'
            info.displayinfo('4', summary, description, keyword, iphase + 1)
        elif str(int(line[0])) not in material_properties.keys():
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + 'Check {}th material phase.'
            info.displayinfo('4', summary, description, keyword, iphase + 1)
        elif str(int(line[0])) in phase_n_clusters.keys():
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + 'Check {}th material phase.'
            info.displayinfo('4', summary, description, keyword, iphase + 1)
        phase_n_clusters[str(int(line[0]))] = int(line[1])
    return phase_n_clusters
# =============================================================================
def read_cluster_analysis_scheme(file, file_path, keyword, material_phases,
                                 clustering_features):
    """Read cluster analysis scheme.

    The specification of the data associated with the cluster analysis scheme
    has the following input data file syntax:

    .. code-block:: text

       Clustering_Analysis_Scheme
       < phase_id > < clustering_type >
           base_clustering
           < clustering_algorithm_id > < feature_id > [< feature_id >]
           adaptive_clustering
           < clustering_algorithm_id > < feature_id > [< feature_id >]
           adaptivity_parameters < adapt_criterion_id > < adapt_type_id >
           < adapt_parameter_name > < value >
           < adapt_parameter_name > < value >
       < phase_id > < clustering_type >
       ...

    where `phase_id` (int) is the material identifier, `clustering_type` is the
    clustering type ({static, adaptive}), `clustering_algorithm_id` (int) is
    the clustering algorithm identifier, `feature_id` (int) is the clustering
    feature identifier, `adapt_criterion_id`(int) is the clustering
    adaptivity criterion identifier, `adapt_type_id` (int) is the adaptive
    cluster-reduced material phase type identifier, and `adapt_parameter_name`
    is the adaptive parameter name.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword : str
        Keyword.
    material_phases : list[str]
        RVE material phases labels (str).
    clustering_features : list[str]
        Available clustering features.

    Returns
    -------
    clustering_type : dict
        Clustering type (item, {'static', 'adaptive'}) of each material phase
        (key, str).
    base_clustering_scheme : dict
        Prescribed base clustering scheme (item, numpy.ndarray of shape
        (n_clusterings, 3)) for each material phase (key, str). Each row is
        associated with a unique clustering characterized by a clustering
        algorithm (col 1, int), a list of features (col 2, list[int]) and a
        list of the features data matrix' indexes (col 3, list[int]).
    adaptive_clustering_scheme : dict
        Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
        (n_clusterings, 3)) for each material phase (key, str). Each row is
        associated with a unique clustering characterized by a clustering
        algorithm (col 1, int), a list of features (col 2, list[int]) and a
        list of the features data matrix' indexes (col 3, list[int]).
    adapt_criterion_data : dict
        Clustering adaptivity criterion (item, dict) associated with each
        material phase (key, str). This dictionary contains the adaptivity
        criterion to be used and the required parameters.
    adaptivity_type : dict
        Clustering adaptivity type (item, dict) associated with each material
        phase (key, str). This dictionary contains the adaptivity type to be
        used and the required parameters.
    adaptivity_control_feature : dict
        Clustering adaptivity control feature (item, str) associated with each
        material phase (key, str).
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find keyword line number
    keyword_line_number = searchkeywordline(file, keyword)
    # Initialize line number
    line_number = keyword_line_number
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize clustering type
    clustering_type = {}
    # Initialize base clustering scheme
    base_clustering_scheme = {}
    # Initialize adaptive clustering scheme and adaptivity related dictionaries
    adaptive_clustering_scheme = {}
    adapt_criterion_data = {}
    adaptivity_type = {}
    adaptivity_control_feature = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material phases
    for i in range(len(material_phases)):
        # Increment line number and read line
        line_number += 1
        line = linecache.getline(file_path, line_number)
        is_empty_line = not bool(line.split())
        # Read material phase and clustering type
        if is_empty_line:
            summary = 'Missing clustering scheme for material phase'
            description = 'The keyword - {} - is not properly defined in the '\
                + 'input data file.' + '\n' \
                + indent + 'The clustering scheme must be specified for all ' \
                + 'material phases.'
            info.displayinfo('4', summary, description, keyword)
        else:
            line = line.split()
        if line[0] not in material_phases:
            summary = 'Unknown material phase'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.' + '\n' \
                + indent + 'Unknown material phase.'
            info.displayinfo('4', summary, description, keyword)
        else:
            if len(line) == 1:
                mat_phase = line[0]
                ctype = 'static'
            elif line[1] not in ['static', 'adaptive']:
                summary = 'Unknown clustering type'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + indent + 'Unknown clustering type.'
                info.displayinfo('4', summary, description, keyword)
            else:
                mat_phase = line[0]
                ctype = line[1]
        # Initialize material phase base clustering
        base_clustering_scheme[mat_phase] = np.full((0, 3), '', dtype=object)
        # Store material phase clustering type
        clustering_type[mat_phase] = ctype
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment line number and read line
        line_number += 1
        line = linecache.getline(file_path, line_number)
        is_empty_line = not bool(line.split())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read base clustering scheme
        if is_empty_line or line.split()[0] != 'base_clustering':
            summary = 'Missing base clustering scheme'
            description = 'The keyword - {} - is not properly defined ' \
                + 'in the input data file.' + '\n' \
                + indent + 'Missing base clustering scheme of material ' \
                + 'phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)
        else:
            # Read base clustering scheme
            base_clustering_scheme[mat_phase], line_number = \
                read_clustering_scheme(file, file_path, line_number)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Skip to the next material phase if static cluster-reduced material
        # phase, otherwise read following line
        if ctype == 'static':
            # Get static cluster-reduced material phase valid clustering
            # algorithms
            valid_algorithms = SCRMP.get_valid_clust_algs()
            # Check validity of prescribed base clustering scheme
            check_clustering_scheme(
                mat_phase, base_clustering_scheme[mat_phase], valid_algorithms,
                clustering_features)
            # Skip to the next material phase
            continue
        else:
            # Initialize material phase adaptivity related dictionaries
            adapt_criterion_data[mat_phase] = {}
            adaptivity_type[mat_phase] = {}
            # Read following line
            line_number += 1
            line = linecache.getline(file_path, line_number)
            is_empty_line = not bool(line.split())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read adaptive clustering scheme
        if is_empty_line or line.split()[0] != 'adaptive_clustering':
            summary = 'Missing adaptive clustering scheme'
            description = 'The keyword - {} - is not properly defined ' \
                + 'in the input data file.' + '\n' \
                + indent + 'Missing adaptive clustering scheme of adaptive ' \
                + 'material phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)
        else:
            # Read adaptive clustering scheme
            adaptive_clustering_scheme[mat_phase], line_number = \
                read_clustering_scheme(file, file_path, line_number)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment line number and read line
        line_number += 1
        line = linecache.getline(file_path, line_number)
        is_empty_line = not bool(line.split())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read adaptivity parameters
        if is_empty_line or line.split()[0] != 'adaptivity_parameters':
            summary = 'Missing adaptivity parameters of clustering scheme'
            description = 'The keyword - {} - is not properly defined ' \
                + 'in the input data file.' + '\n' \
                + indent + 'Missing adaptivity parameters of clustering ' \
                + 'scheme of adaptive material phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)

        else:
            # Read adaptivity criterion and adaptivity type
            line = line.split()
            if line[1] not in \
                    AdaptivityManager.get_adaptivity_criterions().keys():
                summary = 'Unknown adaptivity criterion'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + indent + 'Unknown adaptivity criterion on clustering ' \
                    + 'scheme of adaptive material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            elif line[2] not in CRVE.get_crmp_types().keys():
                summary = 'Unknown adaptivity type'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + indent + 'Unknown adaptivity type on clustering ' \
                    + 'scheme of adaptive material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            else:
                # Read adaptivity criterion
                adapt_criterion_id = line[1]
                adapt_crit = AdaptivityManager.get_adaptivity_criterions()[
                    adapt_criterion_id]
                adapt_criterion_data[mat_phase]['criterion'] = adapt_crit
                # Read adaptivity type
                adapt_type_id = line[2]
                adapt_type = CRVE.get_crmp_types()[adapt_type_id]
                adaptivity_type[mat_phase]['adapt_type'] = adapt_type
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get adaptive cluster-reduced material phase valid clustering
        # algorithms
        valid_algorithms = adapt_type.get_valid_clust_algs()
        # Check validity of prescribed base clustering scheme
        check_clustering_scheme(
            mat_phase, adaptive_clustering_scheme[mat_phase], valid_algorithms,
            clustering_features)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get mandatory and optional adaptivity criterion parameters
        macp, oacp = adapt_crit.get_parameters()
        # Get mandatory and optional adaptivity type parameters
        matp, oatp = adapt_type.get_adaptivity_type_parameters()
        # Collect all mandatory and optional adaptivity parameters
        madapt_parameters = {**macp, **matp}
        oadapt_parameters = {**oacp, **oatp}
        # Set the optional parameters default values by default
        for parameter in oadapt_parameters:
            # Optional adaptivity criterion parameters
            if parameter in oacp.keys():
                # Store adaptivity criterion parameter
                adapt_criterion_data[mat_phase][parameter] = \
                    get_formatted_parameter(parameter, oacp[parameter])
            # Optional adaptivity type parameters
            else:
                # Store adaptivity type parameter
                adaptivity_type[mat_phase][parameter] = \
                    get_formatted_parameter(parameter, oatp[parameter])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read following line
        line = linecache.getline(file_path, line_number + 1)
        is_empty_line = not bool(line.split())
        # Check for adaptivity parameters specifications
        is_adapt_parameter = False
        if is_empty_line:
            pass
        elif line.split()[0] in [*madapt_parameters.keys(),
                                 *oadapt_parameters.keys(),
                                 'adaptivity_control_feature']:
            is_adapt_parameter = True
            parameter = line.split()[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        while is_adapt_parameter:
            # Increment line number and read line
            line_number += 1
            line = linecache.getline(file_path, line_number).split()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check parameter specification
            if len(line) < 2:
                summary = 'Missing adaptivity parameter value'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + indent + 'Missing value of adaptivity parameter on ' \
                    + 'clustering scheme of adaptive material phase {}.'
                info.displayinfo('4', summary, description, keyword, mat_phase)
            # Get adaptivity parameter
            if parameter in macp.keys():
                # Store adaptivity criterion parameter
                adapt_criterion_data[mat_phase][parameter] = \
                    get_formatted_parameter(parameter, line[1],
                                            etype=type(macp[parameter]))
            elif parameter in matp.keys():
                # Store adaptivity type parameter
                adaptivity_type[mat_phase][parameter] = \
                    get_formatted_parameter(parameter, line[1],
                                            etype=type(matp[parameter]))
            elif parameter in oacp.keys():
                # Get parameter value
                value = get_formatted_parameter(parameter, line[1])
                # Store adaptivity criterion parameter
                if isinstance(value, type(oacp[parameter])):
                    adapt_criterion_data[mat_phase][parameter] = \
                        get_formatted_parameter(parameter,
                                                type(oacp[parameter])(line[1]))
                else:
                    summary = 'Invalid adaptivity parameter specification'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Check adaptivity parameter {} on ' \
                        + 'clustering scheme of adaptive material phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     parameter, mat_phase)
            elif parameter in oatp.keys():
                # Get parameter value
                value = get_formatted_parameter(parameter, line[1])
                # Store adaptivity type parameter
                if isinstance(value, type(oatp[parameter])):
                    adaptivity_type[mat_phase][parameter] = \
                        get_formatted_parameter(parameter,
                                                type(oatp[parameter])(line[1]))
                else:
                    summary = 'Invalid adaptivity parameter specification'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Check adaptivity parameter {} on ' \
                        + 'clustering scheme of adaptive material phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     parameter, mat_phase)
            elif parameter == 'adaptivity_control_feature':
                # Store adaptivity control feature
                if mat_phase in adaptivity_control_feature.keys():
                    summary = 'Multiple adaptivity control features'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Only one adaptivity control feature ' \
                        + 'can be prescribed in the clustering scheme ' \
                        + 'of adaptive' + '\n' \
                        + indent + 'material phase {}.'
                    info.displayinfo('4', summary, description, keyword,
                                     mat_phase)
                else:
                    adaptivity_control_feature[mat_phase] = line[1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Read following line
            line = linecache.getline(file_path, line_number + 1)
            is_empty_line = not bool(line.split())
            # Check for adaptivity parameters specifications. If there are no
            # adaptivity parameters specifications, skip to the next material
            # phase
            is_adapt_parameter = False
            if is_empty_line:
                if i != range(len(material_phases))[-1]:
                    summary = 'Missing clustering scheme'
                    description = 'The keyword - {} - is not properly ' \
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'The clustering scheme must be specified ' \
                        + 'for all material phases.'
                    info.displayinfo('4', summary, description, keyword)
            elif line.split()[0] in [*madapt_parameters.keys(),
                                     *oadapt_parameters.keys(),
                                     'adaptivity_control_feature']:
                is_adapt_parameter = True
                parameter = line.split()[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all the mandatory adaptivity criterion parameters have been
        # prescribed
        for parameter in macp.keys():
            if parameter not in adapt_criterion_data[mat_phase].keys():
                summary = 'Missing mandatory adaptivity criterion parameter'
                description = 'The keyword - {} - is not properly ' \
                    + 'defined in the input data file.' + '\n' \
                    + indent + 'Missing mandatory parameter - {} - of ' \
                    + 'adaptive material phase {}.'
                info.displayinfo('4', summary, description, keyword,
                                 parameter, mat_phase)
        # Check if all the mandatory adaptivity type parameters have been
        # prescribed
        for parameter in matp.keys():
            if parameter not in adaptivity_type[mat_phase].keys():
                summary = 'Missing mandatory adaptivity type parameter'
                description = 'The keyword - {} - is not properly ' \
                    + 'defined in the input data file.' + '\n' \
                    + indent + 'Missing mandatory parameter - {} - of ' \
                    + 'adaptive material phase {}.'
                info.displayinfo('4', summary, description, keyword,
                                 parameter, mat_phase)
        # Check if adaptivity control feature has been prescribed
        if mat_phase not in adaptivity_control_feature.keys():
            summary = 'Missing adaptivity control feature'
            description = 'The keyword - {} - is not properly ' \
                + 'defined in the input data file.' + '\n' \
                + indent + 'Missing adaptivity control feature of ' \
                + 'adaptive material phase {}.'
            info.displayinfo('4', summary, description, keyword, mat_phase)
        # Set default values for all the optional adaptivity criterion
        # parameters that have not been prescribed
        for parameter in oacp.keys():
            if parameter not in adapt_criterion_data[mat_phase].keys():
                adapt_criterion_data[mat_phase][parameter] = oacp[parameter]
        # Set default values for all the optional adaptivity type parameters
        # that have not been prescribed
        for parameter in oatp.keys():
            if parameter not in adaptivity_type[mat_phase].keys():
                adaptivity_type[mat_phase][parameter] = oatp[parameter]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if the cluster analysis scheme has been prescribed for all material
    # phases
    if set(base_clustering_scheme.keys()) != set(material_phases):
        summary = 'Missing clustering scheme'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.' + '\n' \
            + indent + 'The clustering scheme must be specified for all ' \
            + 'material phases.'
        info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return clustering_type, base_clustering_scheme, \
        adaptive_clustering_scheme, adapt_criterion_data, adaptivity_type, \
        adaptivity_control_feature
# =============================================================================
def read_clustering_scheme(file, file_path, line_number):
    """Read material phase's prescribed clustering scheme.

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    line_number : int
        Data file line where the material phase's clustering scheme
        prescription begins.

    Returns
    -------
    clustering_scheme : numpy.ndarray of shape (n_clusterings, 3)
        Clustering scheme stored as a numpy.ndarray of shape
        (n_clusterings, 3). Each row is associated with a unique clustering
        characterized by a clustering algorithm (col 1, int), a list of
        clustering features (col 2, list[int]), and a list of the clustering
        features data matrix' indexes (col 3, list[int]).
    line_number : int
        Data file line where the material phase's clustering scheme
        prescription ends.
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize clustering scheme
    clustering_scheme = np.full((0, 3), '', dtype=object)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read line
    line = linecache.getline(file_path, line_number).split()
    # Check number of prescribed clusterings. If not specified, then assume a
    # single clustering solution
    if len(line) > 1:
        if not ioutil.checkposint(line[1]):
            summary = 'Invalid number of clusterings'
            description = 'The number of clusterings of a given clustering ' \
                + 'scheme specified in the input data file' + '\n' \
                + indent + 'is invalid.'
            info.displayinfo('4', summary, description)
        else:
            n_clusterings = int(line[1])
    else:
        n_clusterings = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over prescribed clustering solutions
    for j in range(n_clusterings):
        # Increment line number and read line
        line_number += 1
        line = linecache.getline(file_path, line_number).split()
        # Check clustering solution
        if any([not ioutil.checkposint(x) for x in line]):
            summary = 'Invalid clustering algorithm or feature identifier'
            description = 'A clustering algorithm or feature specified in ' \
                + 'the input data file is invalid.'
            info.displayinfo('4', summary, description)
        # Append clustering solution to clustering scheme
        clustering_scheme = np.append(
            clustering_scheme, np.full((1, 3), '', dtype=object), axis=0)
        # Assemble clustering solution
        clustering_scheme[j, 0] = int(line[0])
        clustering_scheme[j, 1] = list(set([int(x) for x in line[1:]]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return clustering_scheme, line_number
# =============================================================================
def check_clustering_scheme(mat_phase, clustering_scheme, valid_algorithms,
                            valid_features):
    """Check material phase's prescribed clustering scheme.

    Parameters
    ----------
    mat_phase : str
        Material phase label.
    clustering_scheme : numpy.ndarray of shape (n_clusterings, 3)
        Clustering scheme stored as a numpy.ndarray of shape
        (n_clusterings, 3). Each row is associated with a unique clustering
        characterized by a clustering algorithm (col 1, int), a list of
        clustering features (col 2, list[int]), and a list of the clustering
        features data matrix' indexes (col 3, list[int]).
    valid_algorithms : list[str]
        Valid clustering algorithms identifiers (str).
    valid_features : list[str]
        Valid clustering features identifiers (str).
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check validity of prescribed clustering algorithms
    if any([str(x) not in valid_algorithms for x in clustering_scheme[:, 0]]):
        summary = 'Invalid clustering algorithm'
        description = 'A clustering algorithm prescribed in the clustering ' \
            + 'scheme of material phase {} in the' + '\n' \
            + indent + 'input data file is invalid.'
        info.displayinfo('4', summary, description, mat_phase)
    # Check validity of prescribed clustering features
    for j in range(clustering_scheme.shape[0]):
        if any([str(x) not in valid_features
                for x in clustering_scheme[j, 1]]):
            summary = 'Invalid clustering feature'
            description = 'A clustering feature prescribed in the clustering '\
                + 'scheme of material phase {} in the' + '\n' \
                + indent + 'input data file is invalid.'
            info.displayinfo('4', summary, description, mat_phase)
# =============================================================================
def read_adaptivity_frequency(file, file_path, keyword, adapt_material_phases):
    """Read clustering adaptivity frequency.

    The specification of the data associated with the clustering adaptivity
    frequency has the following input data file syntax:

    .. code-block:: text

       Adaptivity_Frequency
       < phase_id > < option >
       < phase_id > < option >

    where `phase_id` (int) is the material identifier and `option` is the
    corresponding clustering adaptivity frequency ({none, all, every < int >})
    with respect to the loading incrementation.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.
    adapt_material_phases : list[str]
        RVE adaptive material phases labels (str).

    Returns
    -------
    clust_adapt_freq : dict, default=None
        Clustering adaptivity frequency (relative to loading incrementation)
        (item, int) associated with each adaptive cluster-reduced material
        phase (key, str).
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find keyword line number
    keyword_line_number = searchkeywordline(file, keyword)
    # Initialize line number
    line_number = keyword_line_number
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize clustering adaptivity frequency
    clust_adapt_freq = {mat_phase: 1 for mat_phase in adapt_material_phases}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read following line
    line = linecache.getline(file_path, line_number + 1)
    is_empty_line = not bool(line.split())
    # Check for adaptivity frequency specifications. If there are no adaptivity
    # frequency specifications, return
    if is_empty_line:
        return clust_adapt_freq
    else:
        line = line.split()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    while True:
        # Increment line number and read line
        line_number += 1
        line = linecache.getline(file_path, line_number).split()
        # Check adaptivity frequency specification
        if len(line) < 2:
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined ' \
                + 'in the input data file.' + '\n' \
                + indent + 'Missing adaptivity frequency specification.'
            info.displayinfo('4', summary, description, keyword)
        elif line[0] not in adapt_material_phases:
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined ' \
                + 'in the input data file.' + '\n' \
                + indent + 'Unknown adaptive material phase.'
            info.displayinfo('4', summary, description, keyword)
        elif line[1] not in ['all', 'none', 'every']:
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined ' \
                + 'in the input data file.' + '\n' \
                + indent + 'Unknown adaptivity frequency option.'
            info.displayinfo('4', summary, description, keyword)
        # Get material phase and adaptivity frequency option
        mat_phase = line[0]
        option = line[1]
        # Set adaptivity frequency
        if option == 'none':
            clust_adapt_freq[mat_phase] = 0
        elif option == 'every':
            # Check option specification
            if len(line) < 3 or not ioutil.checkposint(line[2]):
                summary = 'Invalid keyword specification'
                description = 'The keyword - {} - is not properly defined ' \
                    + 'in the input data file.' + '\n' \
                    + indent + 'Invalid adaptivity frequency option.'
                info.displayinfo('4', summary, description, keyword)
            else:
                clust_adapt_freq[mat_phase] = int(line[2])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read following line
        line = linecache.getline(file_path, line_number + 1)
        is_empty_line = not bool(line.split())
        # Check for adaptivity frequency specifications. If there are no
        # adaptivity frequency specifications, return
        if is_empty_line:
            break
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return clust_adapt_freq
# =============================================================================
def read_rewind_state_parameters(file, file_path, keyword):
    """Read the solution rewind state criterion parameters.

    The specification of the data associated with the solution rewind state
    criterion has the following input data file syntax:

    .. code-block:: text
       :linenos:
       :emphasize-lines: 2

       Analysis_Rewinding
       Analysis_Rewind_State_Criterion < criterion > < parameter_value >
       Analysis_Rewinding_Criterion < criterion > < parameter_value >
       [Max_Number_of_Rewinds < int >]

    where `criterion` (str) is the solution rewind state criterion name and
    `parameter_value` ({int, float}) is the corresponding parameter value.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.

    Returns
    -------
    rewind_state_criterion : tuple
        Rewind state storage criterion [0] and associated parameter [1].
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find keyword line number
    keyword_line_number = searchkeywordline(file, keyword)
    # Get keyword lowercased line
    line = linecache.getline(file_path, keyword_line_number).split()
    line = [x.lower() if not ioutil.checknumber(x) else x for x in line]
    if len(line) == 1:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Missing rewind state storage criterion specification.'
        info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get available rewind state storage criteria and associated default
    # parameters
    available_criteria = RewindManager.get_save_rewind_state_criteria()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get rewind state storage criterion
    if line[1] in available_criteria.keys():
        criterion = line[1]
    else:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Unknown rewind state storage criterion.'
        info.displayinfo('4', summary, description, keyword)
    # Get rewind state storage criterion parameter
    if len(line) > 2:
        # Get specified parameter
        parameter = get_formatted_parameter(
            criterion, line[2], etype=type(available_criteria[criterion]))
        # Set rewind state storage criterion
        rewind_state_criterion = (criterion, parameter)
    else:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Missing rewind state storage criterion parameter.'
        info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return rewind_state_criterion
# =============================================================================
def read_rewinding_criterion_parameters(file, file_path, keyword):
    """Read the solution rewinding criterion parameters.

    The specification of the data associated with the solution rewind criterion
    criterion has the following input data file syntax:

    .. code-block:: text
       :linenos:
       :emphasize-lines: 3

       Analysis_Rewinding
       Analysis_Rewind_State_Criterion < criterion > < parameter_value >
       Analysis_Rewinding_Criterion < criterion > < parameter_value >
       [Max_Number_of_Rewinds < int >]

    where `criterion` (str) is the solution rewind criterion name and
    `parameter_value` ({int, float}) is the corresponding parameter value.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.

    Returns
    -------
    rewinding_criterion : tuple, default=None
        Rewinding criterion [0] and associated parameter [1].
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find keyword line number
    keyword_line_number = searchkeywordline(file, keyword)
    # Get keyword lowercased line
    line = linecache.getline(file_path, keyword_line_number).split()
    line = [x.lower() if not ioutil.checknumber(x) else x for x in line]
    if len(line) == 1:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Missing rewinding criterion specification.'
        info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get available rewinding criteria and associated default parameters
    available_criteria = RewindManager.get_rewinding_criteria()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get rewinding criterion
    if line[1] in available_criteria.keys():
        criterion = line[1]
    else:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Unknown rewinding criterion.'
        info.displayinfo('4', summary, description, keyword)
    # Get rewinding criterion parameter
    if len(line) > 2:
        # Get specified parameter
        parameter = get_formatted_parameter(
            criterion, line[2], etype=type(available_criteria[criterion]))
        # Set rewinding criterion
        rewinding_criterion = (criterion, parameter)
    else:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Missing rewinding criterion parameter.'
        info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return rewinding_criterion
# =============================================================================
def read_discretization_file_path(file, file_path, keyword, valid_exts,
                                  discret_file_dir=None):
    """Read spatial discretization file path.

    The specification of the spatial discretization file path has the following
    input data file syntax:

    .. code-block:: text

       Discretization_File
       < path >

    where `path` is the path of the spatial discretization file.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.
    valid_exts : tuple[str]
        Valid extensions of spatial discretization file.
    discret_file_dir : str, default=None
        Spatial discretization file directory path.

    Returns
    -------
    discret_file_path : str
        Spatial discretization file path.
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    line_number = searchkeywordline(file, keyword) + 1
    discret_file_path = linecache.getline(file_path, line_number).strip()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if spatial discretization file directory path was provided
    if discret_file_dir is not None:
        # If spatial discretization file absolute path is not provided, then
        # build it from provided relative path
        if not os.path.isfile(discret_file_path):
            discret_file_path = discret_file_dir + discret_file_path
            # Get spatial discretization file absolute path
            discret_file_path = os.path.abspath(discret_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not os.path.isfile(discret_file_path):
        summary = 'Missing spatial discretization file'
        description = 'The spatial discretization file specified under ' \
            + 'the keyword - {} - could' + '\n' \
            + indent + 'not be found:' + '\n\n' \
            + indent + '{}'
        info.displayinfo('4', summary, description, keyword,
                         discret_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    format_exts = ['.npy']
    if os.path.splitext(os.path.basename(discret_file_path))[-1] \
            in format_exts:
        if not os.path.splitext(os.path.splitext(os.path.basename(
                discret_file_path))[0])[-1] in valid_exts:
            summary = 'Invalid spatial discretization file extension'
            description = 'The spatial discretization file specified under ' \
                + 'the keyword - {} - does' + '\n' \
                + indent + 'not have a valid extension.'
            info.displayinfo('4', summary, description, keyword)
    else:
        if not os.path.splitext(os.path.basename(
                discret_file_path))[-1] in valid_exts:
            summary = 'Invalid spatial discretization file extension'
            description = 'The spatial discretization file specified under ' \
                + 'the keyword - {} - does' + '\n' \
                + indent + 'not have a valid extension.'
            info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return os.path.abspath(discret_file_path)
# =============================================================================
def read_rve_dimensions(file, file_path, keyword, n_dim):
    """Read RVE dimensions (size length along each spatial dimension).

    The specification of the data associated with the RVE dimensions has the
    following input data file syntax:

    *2D Problem:*

    .. code-block:: text

       RVE_Dimensions
       < dim1_size > < dim2_size >

    where `dimX_size` is the size length along dimension X.

    ----

    *3D Problem:*

    .. code-block:: text

       RVE_Dimensions
       < dim1_size > < dim2_size > < dim3_size >

    where `dimX_size` is the size length along dimension X.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.
    n_dim : int
        Problem number of spatial dimensions.

    Returns
    -------
    rve_dims : list[float]
        RVE size in each dimension.
    """
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number + 1).split()
    if line == '':
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    elif len(line) != n_dim:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    for i in range(n_dim):
        if not ioutil.checknumber(line[i]) or float(line[i]) <= 0:
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.'
            info.displayinfo('4', summary, description, keyword)
    rve_dims = list()
    for i in range(n_dim):
        rve_dims.append(float(line[i]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return rve_dims
# =============================================================================
def read_self_consistent_scheme(file, file_path, keyword, strain_formulation):
    """Read self-consistent scheme and associated parameters.

    The specification of the data associated with the self-consistent scheme
    has the following input data file syntax:

    .. code-block:: text

       Self_Consistent_Scheme < method > [ < n_parameters > ]
       [ < parameter_1_name > < value > ]
       [ < parameter_2_name > < value > ]
       ...

    where `method` (str) is the self-consistent scheme strategy to update the
    reference material properties, `n_parameters` (int) is the number of
    self-consistent scheme parameters, and `parameter_X_name` (str) is the
    self-consistent scheme parameter name.

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.

    Returns
    -------
    self_consistent_scheme : {'none', 'regression'}
        Self-consistent scheme to update the elastic reference material
        properties.
    scs_parameters : {dict, None}
        Self-consistent scheme parameters (key, str; item, {int, float, bool}).
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read keyword line
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number).split()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get available self-consistent schemes
    available_scs = \
        ElasticReferenceMaterial.get_available_scs(strain_formulation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if len(line) == 1 or len(line) > 3:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined in the ' \
            + 'input data file.'
        info.displayinfo('4', summary, description, keyword)
    elif str(line[1]) not in available_scs:
        summary = 'Invalid keyword specification'
        description = 'The keyword - {} - is not properly defined ' \
            + 'in the input data file.' + '\n' \
            + indent + 'Unknown self-consistent scheme.'
        info.displayinfo('4', summary, description, keyword)
    else:
        self_consistent_scheme = str(line[1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self-consistent scheme parameters
    if len(line) == 3:
        if not ioutil.checkposint(line[2]):
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in the '\
                + 'input data file.' + '\n' \
                + indent + 'Invalid number of self-consistent scheme ' \
                + 'parameters.'
            info.displayinfo('4', summary, description, keyword)
        else:
            # Get number of self-consistent parameters
            n_parameters = int(line[2])
            # Initialize self-consistent scheme parameters
            scs_parameters = {}
            # Loop over self-consistent scheme parameters
            line_number = keyword_line_number + 1
            for i in range(n_parameters):
                # Get self-consistent scheme parameter specification line
                parameter_line = \
                    linecache.getline(file_path, line_number + i).split()
                # Get self-consistent scheme parameter
                if not parameter_line:
                    summary = 'Missing self-consistent scheme parameter'
                    description = 'The keyword - {} - is not properly '\
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Missing specification of self-consistent '\
                        + 'scheme parameter.'
                    info.displayinfo('4', summary, description, keyword)
                elif len(parameter_line) != 2:
                    summary = 'Invalid self-consistent scheme parameter'
                    description = 'The keyword - {} - is not properly '\
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Invalid specification of self-consistent '\
                        + 'scheme parameter.'
                    info.displayinfo('4', summary, description, keyword)
                elif not ioutil.checkvalidname(parameter_line[0]):
                    summary = 'Invalid self-consistent scheme parameter'
                    description = 'The keyword - {} - is not properly '\
                        + 'defined in the input data file.' + '\n' \
                        + indent + 'Invalid self-consistent scheme parameter '\
                        + 'name.'
                    info.displayinfo('4', summary, description, keyword)
                else:
                    scs_parameters[str(parameter_line[0])] = \
                        get_formatted_parameter(str(parameter_line[0]),
                                                parameter_line[1])
    else:
        scs_parameters = {'E_init': 'init_eff_tangent',
                          'v_init': 'init_eff_tangent'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all reference material isotropic elastic properties were
    # prescribed
    if scs_parameters is not None:
        param_names = scs_parameters.keys()
        if ('E_init' in param_names and 'v_init' not in param_names) \
                or ('E_init' not in param_names and 'v_init' in param_names):
            summary = 'Missing reference material elastic property'
            description = 'The keyword - {} - is not properly '\
                + 'defined in the input data file.' + '\n' \
                + indent + 'Both elastic properties of the elastic ' \
                + 'reference material must be prescribed.'
            info.displayinfo('4', summary, description, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return self_consistent_scheme, scs_parameters
# =============================================================================
def read_vtk_options(file, file_path, keyword, keyword_line_number):
    """Read VTK output options.

    The specification of the data associated with the VTK output options has
    the following input data file syntax:

    .. code-block:: text

       VTK_Output [ < option_1 > < option_2 > ... ]

    where `option_X` is option of the VTK output (e.g., ascii, every < int >,
    all_variables).

    ----

    Parameters
    ----------
    file : file
        Data file.
    file_path : str
        Data file path.
    keyword: str
        Keyword.
    keyword_line_number: int
        Keyword line number.

    Returns
    -------
    vtk_format : {'ascii', 'binary'}
        VTK format.
    vtk_inc_div : int
        VTK output increment divider.
    vtk_vars : {'all', 'common'}
        VTK output constitutive state variables.
    """
    line = linecache.getline(file_path, keyword_line_number).split()
    line = [x.lower() if not ioutil.checknumber(x) else x for x in line]
    if 'binary' in line:
        raise RuntimeError('This VTK format has not been implemented yet.')
    else:
        vtk_format = 'ascii'
    if 'every' in line:
        if not ioutil.checkposint(line[line.index('every') + 1]):
            summary = 'Invalid keyword specification'
            description = 'The keyword - {} - is not properly defined in ' \
                + 'the input data file.'
            info.displayinfo('4', summary, description, keyword)
        vtk_inc_div = int(line[line.index('every') + 1])
    else:
        vtk_inc_div = 1
    if 'common_variables' in line:
        vtk_vars = 'common'
    else:
        vtk_vars = 'all'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return vtk_format, vtk_inc_div, vtk_vars
