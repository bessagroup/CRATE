
Run a benchmark
===============

To have a first quick look at CRATE's essential performance, give it a go by simply going through the following steps:

1. **Install** CRATE from source (check :ref:`label_installation_source`);

2. From the CRATE root directory, **run** the following command:

.. code-block::

    python3 benchmarks/run_crate_benchmark.py

3. **Sit back and watch** how CRATE performs a multi-scale analysis of an heterogeneous material;

4. **Explore** the results of the CRATE simulation stored in the same directory as the corresponding input data file (see code snippet below).

.. note::

   A description of the CRATE simulation output files is provided in :doc:`Step 4 <../basic_usage/step4_post_processing>`!

----

The module :code:`benchmarks/run_crate_benchmark.py` is meant to illustrate how a CRATE simulation can be performed **in a Python environment** by running one of the benchmarks made available with the project (see :doc:`Benchmarks<../validation/benchmarks>`). Different benchmark simulations can be tested by specifying a different **input data file** in the following module line:

.. code-block:: python

    # benchmarks/run_crate_benchmark.py

    ...

    # Set benchmark input data file
    benchmark_file = 'infinitesimal_strains/2d/sca/example_1/uniaxial_tension/' + \
        'example_1_uniaxial_tension_nc54.dat'
    ...
