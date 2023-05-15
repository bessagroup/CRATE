"""Example of CRATE simulation.

This module is meant to illustrate how a CRATE simulation can be performed in
a Python environment by simply executing the command:

| python3 run_crate_benchmark.py

Note that this module works even if CRATE Python package 'cratepy' is not
installed from the Python Package Index (e.g., with pip). However, note that
CRATE third-party package dependencies (e.g., 'numpy') must be installed and
accessible to the Python interpreter.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import os
import pathlib
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1]) + '/src'
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cratepy
# =============================================================================
#
# =============================================================================
# Get benchmarks' data files directory
benchmarks_dir = str(pathlib.Path(__file__).parents[0])
# Set benchmark input data file
benchmark_file = 'infinitesimal_strains/2d/sca/example_1/uniaxial_tension/' + \
    'example_1_uniaxial_tension_nc54.dat'
# Set benchmark input data file path
input_file_path = os.path.join(benchmarks_dir, benchmark_file)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get benchmarks' microstructures directory
microstructure_dir = os.path.join(str(pathlib.Path(__file__).parents[0]),
                                  'microstructures')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Perform CRATE simulation
cratepy.crate_simulation(input_file_path,
                         arg_discret_file_dir=microstructure_dir,
                         is_null_stdout=False)
