
Step 2: Input data
******************

- **Step 2 (Input Data): Set CRATE's user-defined input data file.**

    * The second step is to define a **CRATE's user-defined input data file** (:code:`.dat` file);

    * The input data file contains all the required information about the problem (problem type, material properties, macro-scale loading path, ...) and about the solution procedure (macro-scale loading incrementation, clustering-based domain decomposition, output options, ...). The **spatial discretization file (`.rgmsh` file)** path is provided in the input data file;

    * A complete **CRATE's user-defined input data file** (:code:`.dat` file) template, where each available keyword specification (mandatory or optional) is fully documented, can be found `here <https://github.com/BernardoFerreira/CRATE/blob/master/doc/CRATE_input_data_file.dat>`_. This template file can be copied to a given local simulation directory and be readily used by replacing the `[insert here]` boxes with the suitable specification!

    |

    .. image:: ../../../schematics/doc_CRATE_input_data_file.png
       :width: 80 %
       :align: center

    |
