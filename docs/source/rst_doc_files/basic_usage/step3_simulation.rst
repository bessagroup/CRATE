
Step 3: Simulation
==================

The third step to perform a CRATE simulation is the actual **execution of the CRATE simulator**, providing the input data file defined in :doc:`Step 2 <step2_input_data>`.

Executing the CRATE simulator **in a Python environment** is a single-linear as illustrated in the following (pseudo-)script:

.. code-block:: python

    # python_script.py

    import crate

    # Set input data file path (mandatory)
    input_data_file_path = ...

    # Set spatial discretization file directory path (optional). If the spatial discretization
    # file path specified in the input data file is not absolute, then it is assumed to be
    # relative to the provided spatial discretization file directory
    discret_file_dir = ...

    # Perform CRATE simulation
    crate.crate_simulation(input_data_file_path, arg_discret_file_dir=discret_file_dir)


.. note::
   The CRATE simulator can be also launched **directly from the command line** by executing CRATE's main script (:code:`main.py` file) and passing the input data file as the **first** calling argument as:

    .. code-block::

        python3 CRATE/src/crate/main.py "/path/to/example_input_data_file.dat"

   The optional spatial discretization file directory may be provided as a **second** calling argument after the input data file path.

----

One appealing feature of CRATE is that the simulation execution can be **monitored in real-time** in the terminal console window where the simulator is launched (see figure below). The rich display data includes:

* Problem **launching information** (e.g., CRATE's version, input data file, initial date and time);

* Detailed description of the different **simulation phases**, namely clear starting and ending delimiters and underlying computational steps;

* In the particular phase of the solution of the micro-scale equilibrium problem, the **incremental procedure** is thoroughly detailed, providing the user information about (i) the macro-scale loading path incrementation, (ii) the iterative solution procedure (e.g., iterations, residuals), and (iii) the homogenized strain and stress tensors;

* Problem **ending information** (e.g., input data file, ending date and time, total execution time, execution time summary table);

* In the case of a failed simulation (e.g., maximum level of sub-incrementation, non-convergent solution), **abortion information** is provided.

.. image:: ../../../schematics/doc_CRATE_execution_output.png
   :width: 80 %
   :align: center

|
