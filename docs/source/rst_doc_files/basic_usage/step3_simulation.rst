
Step 3: Simulation
******************

- **Step 3: (Execution) Run CRATE simulator.**

    * The third step consists in **running CRATE to perform the numerical simulation**;

    * Running CRATE is a single-liner as illustrated in the following Python (pseudo-)script:

      .. code-block:: python

        import crate

        # Set input data file path (mandatory)
        input_data_file_path = ...

        # Set spatial discretization file directory path (optional). If the spatial discretization
        # file path specific in the input data file is not absolute, then it is assumed to be
        # relative to the provided spatial discretization file directory
        discret_file_dir = ...

        # Perform numerical simulation
        crate.crate_simulation(input_data_file_path, discret_file_dir=discret_file_dir)

    * CRATE can also be launched directly from the command line by executing the main script and providing the required inputs as arguments as:

    .. code-block::

        python3 CRATE/src/crate/main.py ‘/path/to/example_input_data_file.dat’ ‘/path/to/discretization/file/directory/’

    * The program execution can be **monitored in real-time** in the terminal console window where the previous script is run. Display data includes program launching information, a detailed description of the different simulation phases, and a execution summary when the program is successfully completed.

    |

    .. image:: ../../../schematics/doc_CRATE_execution_output.png
       :width: 80 %
       :align: center
    |
