"""Green operator and cluster interaction tensors related operations.

This module includes several procedures required to compute the cluster
interaction tensors, namely (1) the frequency discretization of the spatial
domain and (2) the computation of the Green operator, as well as the assembly
of the global cluster interaction matrix.

Functions
---------
set_discrete_freqs
    Perform frequency discretization of the spatial domain.
gop_material_independent_terms
    Compute Green operator material independent terms in frequency domain.
assemble_cit
    Assemble global cluster interaction matrix.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import itertools as it
# Third-party
import numpy as np
import numpy.matlib
# Local
import tensor.matrixoperations as mop
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
#                                                          Discrete frequencies
# =============================================================================
def set_discrete_freqs(n_dim, rve_dims, n_voxels_dims):
    """Perform frequency discretization of the spatial domain.

    Perform frequency discretization by setting the spatial discrete
    frequencies (rad/m) for each dimension.

    *2D case*:

    .. math::

       \\boldsymbol{\\zeta}_{s_{1}, s_{2}} =
          \\left( \\dfrac{2\\pi}{(l_{\\mathrm{RVE}})_{1}} s_{1}, \\,
          \\dfrac{2\\pi}{(l_{\\mathrm{RVE}})_{2}} s_{2} \\right) \\, ,

    .. math::

       s_{i}=0, 1, \\dots, n_{i}-1 \\, , \\quad i=1,2 \\, .

    where :math:`\\boldsymbol{\\zeta}_{s_{1}, s_{2}} \\equiv
    \\boldsymbol{\\zeta}(s_{1}, s_{2})` denotes a sampling angular frequency,
    :math:`(l_{\\mathrm{RVE}})_{i}` is the RVE size in the :math:`i` th
    dimension, and :math:`n_{i}` is the number of voxels in the :math:`i` th
    dimension.

    ----

    *3D case*:

    .. math::

       \\boldsymbol{\\zeta}_{s_{1}, s_{2}, s_{3}} =
          \\left( \\dfrac{2\\pi}{(l_{\\mathrm{RVE}})_{1}} s_{1}, \\,
          \\dfrac{2\\pi}{(l_{\\mathrm{RVE}})_{2}} s_{2}, \\,
          \\dfrac{2\\pi}{(l_{\\mathrm{RVE}})_{3}} s_{3} \\right) \\, ,

    .. math::

       s_{i}=0, 1, \\dots, n_{i}-1 \\, , \\quad i=1,2,3 \\, .

    where :math:`\\boldsymbol{\\zeta}_{s_{1}, s_{2}, s_{3}} \\equiv
    \\boldsymbol{\\zeta}(s_{1}, s_{2}, s_{3})` denotes a sampling angular
    frequency, :math:`(l_{\\mathrm{RVE}})_{i}` is the RVE size in the
    :math:`i` th dimension, and :math:`n_{i}` is the number of voxels in the
    :math:`i` th dimension.

    ----

    Parameters
    ----------
    n_dim : int
        Problem number of spatial dimensions.
    rve_dims : list
        RVE size in each dimension.
    n_voxels_dims : list[int]
        Number of voxels in each dimension of the regular grid (spatial
        discretization of the RVE).

    Returns
    -------
    freq_dims : list[numpy.ndarray (1d)]
        List containing the sample frequencies (numpy.ndarray (1d) of shape
        (n_voxels_dim,)) for each dimension.
    """
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i],
                                                 sampling_period))
    # Return
    return freqs_dims
#
#                                                                Green operator
# =============================================================================
def gop_material_independent_terms(strain_formulation, problem_type, rve_dims,
                                   n_voxels_dims):
    """Compute Green operator material independent terms in frequency domain.

    The Green operator is here assumed to be associated with an isotropic
    elastic reference material.

    The material independent terms of the Green operator are conveniently
    computed to perform an efficient update of the Green operator if the
    associated reference material elastic properties are updated by any means
    (e.g., self-consistent scheme).

    *Infinitesimal strains*:

    .. math::

       (\\breve{\\Phi}^{0}_{1})_{klij} (\\boldsymbol{\\zeta}) =
       \\dfrac{ \\delta_{ki} \\zeta_{j} \\zeta_{l}
              + \\delta_{kj} \\zeta_{i} \\zeta_{l}
              + \\delta_{li} \\zeta_{j} \\zeta_{k}
              + \\delta_{lj} \\zeta_{i} \\zeta_{k}}
              {||\\boldsymbol{\\zeta}||^{2}} \\, ,

    .. math::

        (\\breve{\\Phi}^{0}_{2})_{klij} (\\boldsymbol{\\zeta}) =
        - \\dfrac{\\zeta_{k}\\zeta_{l}\\zeta_{i}\\zeta_{j}}
                 {||\\boldsymbol{\\zeta}||^{4}} \\, ,

    .. math::

       i,j,k,l =1, \\dots, n_{\\text{dim}} \\, ,

    where :math:`(\\breve{\\Phi}^{0}_{1})_{klij}` and
    :math:`(\\breve{\\Phi}^{0}_{2})_{klij}` are the first and second
    fourth-order Green operator material independent terms, respectively,
    :math:`\\boldsymbol{\\zeta}` is the frequency wave vector,
    :math:`\\delta_{ij}` is the Kronecker delta, and :math:`n_{\\text{dim}}`
    is the number of spatial dimensions.

    ----

    *Finite strains*:

    .. math::

       (\\breve{\\Phi}^{0}_{1})_{klij} (\\boldsymbol{\\zeta}) =
       \\dfrac{ \\delta_{ki} \\zeta_{j} \\zeta_{l}}
              {||\\boldsymbol{\\zeta}||^{2}} \\, , \\qquad
        (\\breve{\\Phi}^{0}_{2})_{klij} (\\boldsymbol{\\zeta}) =
        - \\dfrac{\\zeta_{k}\\zeta_{l}\\zeta_{i}\\zeta_{j}}
                 {||\\boldsymbol{\\zeta}||^{4}} \\, ,

    .. math::

       i,j,k,l =1, \\dots, n_{\\text{dim}} \\, .

    where :math:`(\\breve{\\Phi}^{0}_{1})_{klij}` and
    :math:`(\\breve{\\Phi}^{0}_{2})_{klij}` are the first and second
    fourth-order Green operator material independent terms, respectively,
    :math:`\\boldsymbol{\\zeta}` is the frequency wave vector,
    :math:`\\delta_{ij}` is the Kronecker delta, and :math:`n_{\\text{dim}}`
    is the number of spatial dimensions.

    A detailed description of the computational implementation based on
    Hadamard (element-wise) operations can be found in Section 4.6 of
    Ferreira (2022) [#]_.

    .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
           Optimization of Thermoplastic Blends: Microstructural
           Generation, Constitutive Development and Clustering-based
           Reduced-Order Modeling.* PhD Thesis, University of Porto
           (see `here <https://repositorio-aberto.up.pt/handle/10216/
           146900?locale=en>`_)

    ----

    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    rve_dims : list[float]
        RVE size in each dimension.
    n_voxels_dims : list[int]
        Number of voxels in each dimension of the regular grid (spatial
        discretization of the RVE).

    Returns
    -------
    gop_1_dft_vox : dict
        Regular grid shaped matrix (item, numpy.ndarray) containing each
        fourth-order matricial form component (key, str) of the first Green
        operator material independent term in the frequency domain (discrete
        Fourier transform).
    gop_2_dft_vox : dict
        Regular grid shaped matrix (item, numpy.ndarray) containing each
        fourth-order matricial form component (key, str) of the second Green
        operator material independent term in the frequency domain (discrete
        Fourier transform).
    gop_0_freq_dft_vox : dict
        Regular grid shaped matrix (item, numpy.ndarray) containing each
        fourth-order matricial form component (key, str) of the Green operator
        zero-frequency (material independent) term in the frequency domain
        (discrete Fourier transform).
    """
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        mop.get_problem_type_parameters(problem_type)
    # Set strain/stress components order according to problem strain
    # formulation
    if strain_formulation == 'infinitesimal':
        comp_order = comp_order_sym
    elif strain_formulation == 'finite':
        comp_order = comp_order_nsym
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = set_discrete_freqs(n_dim, rve_dims, n_voxels_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Green operator matricial form components
    comps = list(it.product(comp_order, comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form
    # components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x) - 1
                           for x in list(comps[i][0] + comps[i][1])])
        mf_indexes.append([x for x in [comp_order.index(comps[i][0]),
                                       comp_order.index(comps[i][1])]])
    # Set optimized variables
    var1 = [*np.meshgrid(*freqs_dims, indexing='ij')]
    var2 = dict()
    for fo_idx in fo_indexes:
        if str(fo_idx[1]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[1]], var1[fo_idx[3]])
        if ''.join([str(x) for x in fo_idx]) not in var2.keys():
            var2[''.join([str(x) for x in fo_idx])] = \
                np.multiply(np.multiply(var1[fo_idx[0]], var1[fo_idx[1]]),
                            np.multiply(var1[fo_idx[2]], var1[fo_idx[3]]))
        if strain_formulation == 'infinitesimal':
            if str(fo_idx[1]) + str(fo_idx[2]) not in var2.keys():
                var2[str(fo_idx[1]) + str(fo_idx[2])] = \
                    np.multiply(var1[fo_idx[1]], var1[fo_idx[2]])
            if str(fo_idx[0]) + str(fo_idx[2]) not in var2.keys():
                var2[str(fo_idx[0]) + str(fo_idx[2])] = \
                    np.multiply(var1[fo_idx[0]], var1[fo_idx[2]])
            if str(fo_idx[0]) + str(fo_idx[3]) not in var2.keys():
                var2[str(fo_idx[0]) + str(fo_idx[3])] = \
                    np.multiply(var1[fo_idx[0]], var1[fo_idx[3]])
    if n_dim == 2:
        var3 = np.sqrt(np.add(np.square(var1[0]), np.square(var1[1])))
    else:
        var3 = np.sqrt(np.add(np.add(np.square(var1[0]), np.square(var1[1])),
                              np.square(var1[2])))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Green operator material independent terms
    gop_1_dft_vox = {''.join([str(x+1) for x in idx]):
                     np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    gop_2_dft_vox = {''.join([str(x+1) for x in idx]):
                     np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    gop_0_freq_dft_vox = {''.join([str(x+1) for x in idx]):
                          np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Set optimized variables
        var4 = [fo_idx[0] == fo_idx[2], fo_idx[0] == fo_idx[3],
                fo_idx[1] == fo_idx[3], fo_idx[1] == fo_idx[2]]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]),
                str(fo_idx[1]) + str(fo_idx[2]),
                str(fo_idx[0]) + str(fo_idx[2]),
                str(fo_idx[0]) + str(fo_idx[3])]
        var4 = [fo_idx[0] == fo_idx[2], ]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]), ]
        if strain_formulation == 'infinitesimal':
            var4 += [fo_idx[0] == fo_idx[3], fo_idx[1] == fo_idx[3],
                     fo_idx[1] == fo_idx[2]]
            var5 += [str(fo_idx[1]) + str(fo_idx[2]),
                     str(fo_idx[0]) + str(fo_idx[2]),
                     str(fo_idx[0]) + str(fo_idx[3])]
        # Compute first material independent term of Green operator
        first_term = np.zeros(tuple(n_voxels_dims))
        for j in range(len(var4)):
            if var4[j]:
                first_term = np.add(first_term, var2[var5[j]])
        first_term = np.divide(first_term, np.square(var3),
                               where=abs(var3) > 1e-10)
        gop_1_dft_vox[comp] = copy.copy(first_term)
        # Compute second material independent term of Green operator
        gop_2_dft_vox[comp] = -1.0*np.divide(
            var2[''.join([str(x) for x in fo_idx])],
            np.square(np.square(var3)), where=abs(var3) > 1e-10)
        # Compute Green operator zero-frequency term
        gop_0_freq_dft_vox[comp][tuple(n_dim*(0,))] = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gop_1_dft_vox, gop_2_dft_vox, gop_0_freq_dft_vox
#
#                                    Global cluster interaction matrix assembly
# =============================================================================
def assemble_cit(strain_formulation, problem_type, mat_prop_ref,
                 material_phases, phase_n_clusters, phase_clusters, cit_1_mf,
                 cit_2_mf, cit_0_freq_mf):
    """Assemble global cluster interaction matrix.

    Update the cluster interaction tensors by taking into account the material
    properties of the reference material and assemble them in the global
    cluster interaction matrix.

    The dependency on the material properties of the reference material stems
    from the definition of the Green operator as shown below.

    *Infinitesimal strains*:

    .. math::

        \\breve{\\mathbf{\\Phi}}^{0} (\\boldsymbol{\\zeta}) =
        c_{1}(\\lambda^{0}, \\mu^{0}) \\, \\breve{\\mathbf{\\Phi}}^{0}_{1} +
        c_{2}(\\lambda^{0}, \\mu^{0}) \\, \\breve{\\mathbf{\\Phi}}^{0}_{2}
        \\, ,

    .. math::

        c_{1}(\\lambda^{0}, \\mu^{0}) = \\dfrac{1}{4 \\mu^{0}} \\, , \\qquad
        c_{2}(\\lambda^{0}, \\mu^{0}) = \\dfrac{\\lambda^{0} +
        \\mu^{0}}{\\mu^{0}(\\lambda^{0} + 2 \\mu^{0})}

    where :math:`\\breve{\\mathbf{\\Phi}}^{0}_{1}` and
    :math:`\\breve{\\mathbf{\\Phi}}^{0}_{2}` are the first and second
    fourth-order Green operator material independent terms, respectively,
    and :math:`(\\lambda^{0}, \\mu^{0})` are the reference elastic (isotropic)
    material Lamé parameters.

    ----

    *Finite strains*:

    .. math::

        \\breve{\\mathbf{\\Phi}}^{0} (\\boldsymbol{\\zeta}) =
        c_{1}(\\lambda^{0}, \\mu^{0}) \\, \\breve{\\mathbf{\\Phi}}^{0}_{1} +
        c_{2}(\\lambda^{0}, \\mu^{0}) \\, \\breve{\\mathbf{\\Phi}}^{0}_{2}
        \\, ,

    .. math::

        c_{1}(\\lambda^{0}, \\mu^{0}) = \\dfrac{1}{2 \\mu^{0}} \\, , \\qquad
        c_{2}(\\lambda^{0}, \\mu^{0}) =
        \\dfrac{\\lambda^{0}}{2 \\mu^{0}(\\lambda^{0} + 2 \\mu^{0})}

    where :math:`\\breve{\\mathbf{\\Phi}}^{0}_{1}` and
    :math:`\\breve{\\mathbf{\\Phi}}^{0}_{2}` are the first and second
    fourth-order Green operator material independent terms, respectively,
    and :math:`(\\lambda^{0}, \\mu^{0})` are the reference elastic (isotropic)
    material Lamé parameters.

    ----

    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    mat_prop_ref : dict
        Reference material properties.
    material_phases : list[str]
        RVE material phases labels (str).
    phase_n_clusters : dict
        Number of clusters (item, int) prescribed for each material phase
        (key, str).
    phase_clusters : dict
        Clusters labels (item, list[int]) associated to each material phase
        (key, str).
    cit_1_mf : dict
        Cluster interaction tensors associated with the first Green operator
        material independent term. Each tensor is stored in a dictionary
        (item, dict) for each pair of material phases (key, str), which in turn
        contains the corresponding matricial form (item, numpy.ndarray)
        associated to each pair of clusters (key, str).
    cit_2_mf : dict
        Cluster interaction tensors associated with the second Green operator
        material independent term. Each tensor is stored in a dictionary
        (item, dict) for each pair of material phases (key, str), which in turn
        contains the corresponding matricial form (item, numpy.ndarray)
        associated to each pair of clusters (key, str).
    cit_0_freq_mf : dict
        Cluster interaction tensors associated with the zero-frequency Green
        operator (material independent) term. Each tensor is stored in a
        dictionary (item, dict) for each pair of material phases (key, str),
        which in turn contains the corresponding matricial form
        (item, numpy.ndarray) associated to each pair of clusters (key, str).

    Returns
    -------
    global_cit_mf : numpy.ndarray
        Global cluster interaction matrix. Assembly positions are assigned
        according to the order of material_phases (1st) and phase_clusters
        (2nd).
    """
    # Get problem type parameters
    _, comp_order_sym, comp_order_nsym = \
        mop.get_problem_type_parameters(problem_type)
    # Set strain/stress components order according to problem strain
    # formulation
    if strain_formulation == 'infinitesimal':
        comp_order = comp_order_sym
    elif strain_formulation == 'finite':
        comp_order = comp_order_nsym
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get total number of clusters
    n_total_clusters = sum([phase_n_clusters[mat_phase]
                            for mat_phase in material_phases])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get reference material Young modulus and Poisson ratio
    E_ref = mat_prop_ref['E']
    v_ref = mat_prop_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator's reference material coefficients
    if strain_formulation == 'infinitesimal':
        gop_factor_1 = 1.0/(4.0*miu_ref)
        gop_factor_2 = (lam_ref + miu_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    else:
        gop_factor_1 = 1.0/(2.0*miu_ref)
        gop_factor_2 = lam_ref/(2.0*miu_ref*(lam_ref + 2.0*miu_ref))
    gop_factor_0_freq = numpy.matlib.repmat(
        np.zeros((len(comp_order), len(comp_order))), n_total_clusters,
        n_total_clusters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize global clustering interaction matrices
    dims = (n_total_clusters*len(comp_order), n_total_clusters*len(comp_order))
    global_cit_1_mf = np.zeros(dims)
    global_cit_2_mf = np.zeros(dims)
    global_cit_0_freq_mf = np.zeros(dims)
    # Initialize column cluster index
    jclst = 0
    # Loop over material phases
    for mat_phase_B in material_phases:
        # Loop over material phase B clusters
        for cluster_J in phase_clusters[mat_phase_B]:
            # Initialize row cluster index
            iclst = 0
            # Loop over material phases
            for mat_phase_A in material_phases:
                # Set material phase pair
                mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                # Loop over material phase A clusters
                for cluster_I in phase_clusters[mat_phase_A]:
                    # Set cluster pair
                    cluster_pair = str(cluster_I) + '_' + str(cluster_J)
                    # Set assembling ranges
                    i_init = iclst*len(comp_order)
                    i_end = i_init + len(comp_order)
                    j_init = jclst*len(comp_order)
                    j_end = j_init + len(comp_order)
                    # Assemble cluster interaction tensors
                    global_cit_1_mf[i_init:i_end, j_init:j_end] = \
                        cit_1_mf[mat_phase_pair][cluster_pair]
                    global_cit_2_mf[i_init:i_end, j_init:j_end] = \
                        cit_2_mf[mat_phase_pair][cluster_pair]
                    global_cit_0_freq_mf[i_init:i_end, j_init:j_end] = \
                        cit_0_freq_mf[mat_phase_pair][cluster_pair]
                    # Increment row cluster index
                    iclst = iclst + 1
            # Increment column cluster index
            jclst = jclst + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble global cluster interaction matrix
    global_cit_mf = np.add(
        np.add(np.multiply(gop_factor_1, global_cit_1_mf),
               np.multiply(gop_factor_2, global_cit_2_mf)),
        np.multiply(gop_factor_0_freq, global_cit_0_freq_mf))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return global_cit_mf
