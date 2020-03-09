#
# Write Homogenized Results Output Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | March 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Working with arrays
import numpy as np
#
#                                                      Write homogenized results output file
# ==========================================================================================
# Write the homogenized results to an output file (.hres)
def writeHomResFile(hres_file_path,problem_type,inc,hom_results):
    # Set homogenized results output file header
    header = ['Increment',
              'strain_11','strain_22','strain_33','strain_12','strain_23','strain_13',
              'stress_11','stress_22','stress_33','stress_12','stress_23','stress_13']
    # Set column width
    col_width = max(16,max([len(x) for x in header]) + 2)
    # Open homogenized results output file (.hres) and write file header
    if not os.path.isfile(hres_file_path):
        initHomResFile(hres_file_path,col_width,header)
    # Write increment homogenized results
    writeIncHomRes(hres_file_path,col_width,problem_type,inc,hom_results)
#
#                                                                    Complementary functions
# ==========================================================================================
# Open homogenized results output file (.hres) and write file header
def initHomResFile(hres_file_path,col_width,header):
    # Open homogenized results output file (write mode)
    hres_file = open(hres_file_path,'w')
    # Set homogenized results output file header format structure
    write_list = ['{:>9s}'.format(header[0]) +
                   ''.join([('{:>' + str(col_width) + 's}').format(x) for x in header[1:]])]
    # Write homogenized results output file header
    hres_file.writelines(write_list)
    # Close homogenized results output file
    hres_file.close()
# ------------------------------------------------------------------------------------------
# Write increment homogenized results
def writeIncHomRes(hres_file_path,col_width,problem_type,inc,hom_results):
    # Get homogenized data
    hom_strain = hom_results['hom_strain']
    hom_stress = hom_results['hom_stress']
    if problem_type == 1:
        hom_stress_33 = hom_results['hom_stress_33']
    # Open homogenized results output file (append mode)
    hres_file = open(hres_file_path,'a')
    # Set increment homogenized results format structure
    if problem_type == 1:
        inc_data = [inc,hom_strain[0,0],hom_strain[1,1],0.0,
                        hom_strain[0,1],0.0,0.0,
                        hom_stress[0,0],hom_stress[1,1],hom_stress_33,
                        hom_stress[0,1],0.0,0.0]
    else:
        inc_data = [inc,hom_strain[0,0],hom_strain[1,1],hom_strain[2,2],
                        hom_strain[0,1],hom_strain[1,2],hom_strain[0,2],
                        hom_stress[0,0],hom_stress[1,1],hom_stress[2,2],
                        hom_stress[0,1],hom_stress[1,2],hom_stress[0,2]]
    write_list = ['\n' + '{:>9d}'.format(inc) +
               ''.join([('{:>' + str(col_width) + '.8e}').format(x) for x in inc_data[1:]])]
    # Write increment homogenized results
    hres_file.writelines(write_list)
    # Close homogenized results output file
    hres_file.close()
