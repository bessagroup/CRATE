#
# Links Post-Processing Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the post-processing of results outputed by the
# finite element code Links.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Working with arrays
import numpy as np
# Extract information from path
import ntpath
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Links related procedures
import Links.LinksUtilities as LinksUtil
#
#                                                                 Links '.elavg' output file
# ==========================================================================================
# Get the elementwise average strain tensor components
def getLinksStrainVox(Links_file_path,n_dim,comp_order,n_voxels_dims):
    # Initialize strain tensor
    strain_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    # Set elementwise average output file path and check file existence
    elagv_file_name = ntpath.splitext(ntpath.basename(Links_file_path))[0]
    elagv_file_path = ntpath.dirname(Links_file_path) + '/' + \
                                          elagv_file_name + '/' + elagv_file_name + '.elavg'
    if not os.path.isfile(elagv_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00070',location.filename,location.lineno+1,elagv_file_path)
    # Load elementwise average strain tensor components
    elagv_array = np.genfromtxt(elagv_file_path,autostrip=True)
    # Get Links strain components order
    Links_comp_order_sym,_ = LinksUtil.getLinksCompOrder(n_dim)
    # Loop over Links strain components
    for i in range(len(Links_comp_order_sym)):
        # Get Links strain component
        Links_comp = Links_comp_order_sym[i]
        # Set Links Voigt notation factor
        Voigt_factor = 2.0 if Links_comp[0] != Links_comp[1] else 1.0
        # Store elementwise average strain component
        strain_vox[Links_comp] = \
                        (1.0/Voigt_factor)*elagv_array[i,:].reshape(n_voxels_dims,order='F')
    # Return
    return strain_vox
