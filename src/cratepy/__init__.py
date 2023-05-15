"""CRATE (Clustering-based Nonlinear Analysis of Materials).

CRATE was originally developed by Bernardo P. Ferreira in the context of his
PhD Thesis (see Ferreira (2022) [#]_). CRATE is devised to aid the design and
development of new materials by performing multi-scale nonlinear analyses of
heterogeneous materials through a suitable coupling between first-order
computational homogenization and clustering-based reduced-order modeling.

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
       (see `here <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_)
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
from cratepy import main
from cratepy.main import crate_simulation
from cratepy import clustering
from cratepy import ioput
from cratepy import material
from cratepy import online
from cratepy import optimization
from cratepy import tensor
