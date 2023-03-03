



import sys

sys.path.insert(0, "/home/bernardoferreira/Documents/repositories/CRATE/src/crate")

from crate import crate_simulation

input_file_path = '/home/bernardoferreira/Documents/launch_crate/figures/example_input_data_file.dat'

discret_file_dir = '/home/bernardoferreira/Documents/launch_crate/benchmarks/microstructures'

crate_simulation(input_file_path, discret_file_dir)
