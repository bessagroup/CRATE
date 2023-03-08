"""Testing CRATE execution."""
#
#                                                                       Modules
# =============================================================================
import sys
# Add CRATE directory to PYTHONPATH (search path for modules)
sys.path.insert(
    0, '/home/bernardoferreira/Documents/repositories/CRATE/src/crate')
# Import CRATE main module
from main import crate_simulation
# =============================================================================
#
# =============================================================================
# Set input data file path
input_file_path = '/home/bernardoferreira/Documents/launch_crate/figures/example_input_data_file.dat'
# Set spatial discretization file directory
discret_file_dir = '/home/bernardoferreira/Documents/launch_crate/benchmarks/microstructures'
# Perform CRATE simulation
crate_simulation(input_file_path, discret_file_dir)
