"""CRATE (Clustering-based Nonlinear Analysis of Materials).

CRATE's is a Python program whose initial version (1.0.0) was originally
developed by Bernardo P. Ferreira in the context of his PhD Thesis
(see Ferreira (2022) [#]_). CRATE is devised to aid the design and development
of new materials by performing multi-scale nonlinear analyses of heterogeneous
materials through a suitable coupling between first-order computational
homogenization and clustering-based reduced-order modeling.

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
"""
#
#                                                                       Modules
# =============================================================================
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[0])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from crate import main
from crate import clustering
from crate import ioput
from crate import material
from crate import online
from crate import optimization
from crate import tensor
