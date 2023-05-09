
General workflow
================

CRATE's **workflow** to perform a multi-scale analysis of an heterogeneous material involves **4 different steps**:

1. **[Input data]** Generate a Representative Volume Element (RVE) of the material under analysis (see :doc:`Step 1 <step1_material_model>`);

2. **[Input data]** Setup CRATE's user-defined input data file for the multi-scale simulation (see :doc:`Step 2 <step2_input_data>`);

3. **[Execution]** Run CRATE simulator to solve multi-scale equilibrium problem (see :doc:`Step 3 <step3_simulation>`);

4. **[Output data]** Post-process the multi-scale simulations results (see :doc:`Step 4 <step4_post_processing>`).

As can be seen from the previous steps, CRATE essentially requires **setting up a user-defined input data file**, which contains all the required data about the multi-scale equilibrium problem and the simulation procedure itself, and **executing CRATE simulator**, which carries out the whole simulation procedure automatically and provides the results for post-processing.
