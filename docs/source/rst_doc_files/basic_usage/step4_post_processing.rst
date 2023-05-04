
Step 4: Post-processing
***********************

- **Step 4 (Output): Post-process results.**

    * The fourth-step consists in **post-processing the simulation results**;

    * CRATE generates **several output files** during running time that are collectively stored in a single output directory created in the same path and sharing the same name as the input data file. Among these output files, three are particularly useful:

        - :code:`.screen` file - A log file where all the data displayed in the default standard output device is stored;

        - :code:`.hres` file - A file where the macro-scale material response is stored, i.e., the homogenized stress-strain response of the RVE computed at every macro-scale loading increment (show below);

        - :code:`.efftan` file - A file where the RVE effective material consistent tangent modulus computed at every macro-scale loading increment is stored;

        - :code:`.vti` file - A VTK XML output file associated with a given macro-scale loading increment that allows the RVE relevant physical data to be conveniently analyzed with a suitable visualization software (e.g. `ParaView <https://www.paraview.org/>`_) (show below).

    |

    .. image:: ../../../schematics/doc_CRATE_hres_output.png
       :width: 80 %
       :align: center

    |

    .. image:: ../../../schematics/doc_CRATE_vti_output.png
       :width: 80 %
       :align: center

    |
