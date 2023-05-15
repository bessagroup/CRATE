
Customization
=============

Implementing new features
-------------------------

Although vanilla CRATE provides a fairly reasonable set of essential features (see :doc:`Available features <../basic_usage/available_features>`), it is expected that the interested user may want to **implement specific features** seeking a particular application or **implement new features** in the context of research and development.

Given that CRATE is implemented in Python and following an **object-oriented programming (OOP)** paradigm, several interfaces emerged naturally during the computational design process and are now available to easily accommodate new implementations. An **interface** is intrinsically associated with abstraction and polymorphism in OOP, i.e., it is fundamentally a blueprint to implement new classes that respect a given contract and that can be dynamically treated in a unified sense. This contract establishes that certain attributes and/or methods must be defined by new classes according to their specific type and uniqueness.

Of course that the implementation of new features is **not limited to** the available CRATE interfaces, and other **minor improvements** and/or **major developments** are also possible according with the envisaged research/application goals.

.. note ::
   Implementing new features in CRATE requires a **medium level of expertise** in Python programming under an object-oriented programming approach. There is a vast body of literature on such topic (e.g., `Phillips (2018) <https://www.google.com/books/edition/Python_3_Object_Oriented_Programming/08t1DwAAQBAJ?>`_, `Lott (2019) <https://www.google.com/books/edition/Mastering_Object_Oriented_Python/GF6exgEACAAJ?>`_) as well as open-source web resources (e.g., `Real Python <https://realpython.com/>`_, `Refactoring Guru <https://refactoring.guru/design-patterns/python>`_).

----

Available interfaces
--------------------

Below is a list of some CRATE **available interfaces** alongside a brief description and a hyperlink to the corresponding documentation page:

* **Interface: Direct Numerical Simulation solver** - Implement a direct numerical simulation (DNS) multi-scale method solver to perform the required offline-stage simulations (:doc:`interface documentation <interface_dns_solver>`);

* **Interface: Clustering feature** - Implement a clustering feature required to perform the RVE cluster analysis (:doc:`interface documentation <interface_clustering_feature>`);

* **Interface: Clustering algorithm** - Implement a clustering algorithm to perform the RVE cluster analysis (:doc:`interface documentation <interface_clustering_algorithm>`);

* **Interface: Constitutive model** - Implement a constitutive model to describe the physical behavior of a given material phase (:doc:`interface documentation <interface_constitutive_model>`).

----

Other implementations
---------------------

Below is a description of possible implementations that can be regarded as a minor enrichment of existent CRATE toolkits:

* **Tensor/Matrix operations and procedures**:
    - Additional algebraic tensorial operations and tensorial operators can be implemented in module :py:mod:`cratepy.tensor.tensoroperations`;
    - Additional matricial procedures can be implemented in module :py:mod:`cratepy.tensor.matrixoperations`.

* **Computational solid mechanics procedures**:
    - Additional computations and procedures arising in computational solid mechanics can be implemented in module :py:mod:`cratepy.material.materialoperations`.
