#
# CRVE Online Adaptivity Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the CRVE adaptivity during the clustering-based reduced order model
# solution of the microscale equilibrium problem (online stage).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | October 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Shallow and deep copy operations
import copy
# Date and time
import time
# Defining abstract base classes
from abc import ABC, abstractmethod
# Display messages
import ioput.info as info
#
#                                                            CRVE Online Adaptivity Criteria
# ==========================================================================================
class AdaptiveCriterion(ABC):
    '''CRVE online adaptivity criterion interface (WIP).'''
    @abstractmethod
    def __init__(self):
        '''CRVE online adaptivity criterion constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_target_clusters(self):
        '''Get online adaptivity criterion target clusters.

        Parameters
        ----------

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        '''
        pass
#
#                                                 CRVE Online Adaptive Clustering Refinement
# ==========================================================================================
def adaptive_refinement(crve, target_clusters, cluster_dicts, verbose=False):
    '''Perform CRVE online adaptive clustering refinement.

    Parameters
    ----------
    crve : CRVE
        Cluster-reduced Representative Volume Element.
    target_clusters : list
        List containing the labels (int) of clusters to be refined.
    cluster_dicts : list
        List containing cluster-label-keyd dictionaries (key, str) that will be updated as
        the result of the CRVE online adaptive refinement.
    verbose : bool, default=False
        Enable verbose output.
    '''
    init_time = time.time()
    if verbose:
        info.displayinfo('5', 'A-CRVE clustering (adaptive step ' +
                               str(crve._adaptive_step + 1) + '):')
        info.displayinfo('5', 'A - Building A-CRVE clustering...', 2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform CRVE online adaptive clustering refinement
    adaptive_clustering_map, _ = crve.perform_adaptive_clustering(target_clusters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if verbose:
        a_time = time.time() - init_time
        info.displayinfo('5', 'B - Computing A-CRVE cluster interaction tensors...', 2)
        ref_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute CRVE cluster interaction tensors
    crve.compute_cit(mode='adaptive', adaptive_clustering_map=adaptive_clustering_map)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if verbose:
        b_time = time.time() - ref_time
        info.displayinfo('5', 'C - Updating A-CRVE cluster-related quantities...', 2)
        ref_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over material phases
    for mat_phase in crve._material_phases:
        # Loop over material phase target clusters
        for target_cluster in adaptive_clustering_map[mat_phase].keys():
            # Get list of target's child clusters labels
            child_clusters = adaptive_clustering_map[mat_phase][target_cluster]
            # Loop over cluster-key dictionaries
            for cluster_dict in cluster_dicts:
                # Update variables
                for child_cluster in child_clusters:
                    cluster_dict[str(child_cluster)] = \
                        copy.deepcopy(cluster_dict[target_cluster])
                # Remove target cluster item
                cluster_dict.pop(target_cluster)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if verbose:
        c_time = time.time() - ref_time
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update total amount of time spent in the A-CRVE adaptive procedures
    dtime = time.time() - init_time
    crve.adaptive_time += dtime
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if verbose:
        indent = 10*' '
        info.displayinfo('5', 'Execution times (s):' + '\n\n' +
                              indent + 'Phase      Time(s)       %' + '\n' +
                              indent + 28*'-' + '\n' +
                              indent + '{:<5s}'.format('A') + '{:^18.4e}'.format(a_time) +
                              '{:>5.2f}'.format(a_time/dtime) + '\n' +
                              indent + '{:<5s}'.format('B') + '{:^18.4e}'.format(b_time) +
                              '{:>5.2f}'.format(b_time/dtime) + '\n' +
                              indent + '{:<5s}'.format('C') + '{:^18.4e}'.format(c_time) +
                              '{:>5.2f}'.format(c_time/dtime) + '\n' +
                              indent + 28*'-' + '\n' +
                              indent + '{:^5s}'.format('Total') + '{:^18.4e}'.format(dtime),
                              2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:
        # Print adaptive clustering mapping
        print('\n\n' + '  Adaptive cluster mapping: ')
        for mat_phase in crve._material_phases:
            print('\n    Material phase ' + mat_phase + ':\n')
            for old_cluster in adaptive_clustering_map[mat_phase].keys():
                print('      Old cluster: ' + '{:>4s}'.format(old_cluster) +
                      '  ->  ' +
                      'New clusters: ',
                      adaptive_clustering_map[mat_phase][str(old_cluster)])
