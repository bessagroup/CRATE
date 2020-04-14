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
              'stress_11','stress_22','stress_33','stress_12','stress_23','stress_13',
              'vm_strain','vm_stress',
              'strain_1','strain_2','strain_3','stress_1','stress_2','stress_3']
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
    # When the problem type corresponds to a 2D analysis, build the 3D homogenized strain
    # and stress tensors by considering the appropriate out-of-plane strain and stress
    # components
    out_hom_strain = np.zeros((3,3))
    out_hom_stress = np.zeros((3,3))
    if problem_type == 1:
        out_hom_strain[0:2,0:2] = hom_strain
        out_hom_stress[0:2,0:2] = hom_stress
        out_hom_stress[2,2] = hom_stress_33
    else:
        out_hom_strain[:,:] = hom_strain
        out_hom_stress[:,:] = hom_stress
    # Compute the von Mises equivalent strain
    vm_strain = np.sqrt(2.0/3.0)*np.linalg.norm(out_hom_strain - \
                                               (1.0/3.0)*np.trace(out_hom_strain)*np.eye(3))
    # Compute the von Mises equivalent stress
    vm_stress = (1.0/np.sqrt(2))*np.sqrt((out_hom_stress[0,0] - out_hom_stress[1,1])**2 + \
                                         (out_hom_stress[1,1] - out_hom_stress[2,2])**2 + \
                                         (out_hom_stress[2,2] - out_hom_stress[0,0])**2 + \
                                         6.0*(out_hom_stress[0,1]**2 + \
                                              out_hom_stress[1,2]**2 + \
                                              out_hom_stress[0,2]**2))
    # Compute the eigenstrains (strain_1, strain_2, strain_3)
    eigenstrains = np.sort(np.linalg.eig(out_hom_strain)[0])[::-1]
    # Compute the eigenstresses (stress_1, stress_2, stress_3)
    eigenstresses = np.sort(np.linalg.eig(out_hom_stress)[0])[::-1]
    # Open homogenized results output file (append mode)
    hres_file = open(hres_file_path,'a')
    # Set increment homogenized results format structure
    inc_data = [inc,out_hom_strain[0,0],out_hom_strain[1,1],out_hom_strain[2,2],
                    out_hom_strain[0,1],out_hom_strain[1,2],out_hom_strain[0,2],
                    out_hom_stress[0,0],out_hom_stress[1,1],out_hom_stress[2,2],
                    out_hom_stress[0,1],out_hom_stress[1,2],out_hom_stress[0,2],
                    vm_strain,vm_stress,
                    eigenstrains[0],eigenstrains[1],eigenstrains[2],
                    eigenstresses[0],eigenstresses[1],eigenstresses[2]]
    write_list = ['\n' + '{:>9d}'.format(inc) +
               ''.join([('{:>' + str(col_width) + '.8e}').format(x) for x in inc_data[1:]])]
    # Write increment homogenized results
    hres_file.writelines(write_list)
    # Close homogenized results output file
    hres_file.close()
