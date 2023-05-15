
Installation
============

CRATE is a simple **Python package** (`cratepy <https://pypi.org/project/cratepy/>`_) available from the Python Package Index (`PyPI <https://pypi.org/>`_). This means that running CRATE requires solely a **Python 3 installation** and a few well-known Python packages used for scientific computing and data science.

.. note::
    Whether you are using Linux, MacOS or Windows, Python can be easily installed by following the `Python Getting Started <https://www.python.org/about/gettingstarted/>`_ official webpage. Many other resources are available to help you install and setup Python (e.g., `Real Python <https://realpython.com/installing-python/>`_).

----

Installation from Python Package Index
--------------------------------------

The standard way of installing CRATE is through `pip <https://pip.pypa.io/en/stable/getting-started/>`_, a package manager that installs Python packages **from PyPI**. After `installing pip <https://pip.pypa.io/en/stable/installation/>`_, a package can be installed as described `here <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_ and `here <https://pip.pypa.io/en/stable/getting-started/>`_.

CRATE can be simply installed by running the following pip installation command:

.. code-block::

   pip install -U cratepy

By following this installation option, you will get CRATE most recent available version on PyPI, i.e., the latest distribution version of CRATE's main source code uploaded to PyPI. Note that, besides installing CRATE, pip also installs all the required Python package dependencies automatically.

----

.. _label_installation_source:

Installation from source
------------------------

To install CRATE **from the associated GitHub repository**, follow the steps below:

* Clone `CRATE GitHub repository <https://github.com/bessagroup/CRATE>`_ into a local directory (check `git documentation <https://git-scm.com/docs/git-clone>`_ for details) by running the following command:

    .. code-block::

       git clone git@github.com:bessagroup/CRATE.git

* In the cloned repository root directory, install CRATE by running the following pip installation command (check pip `regular installation <https://pip.pypa.io/en/stable/topics/local-project-installs/#regular-installs>`_ from local project for details):

    .. code-block::

       pip install .

.. note::
   From a development point of view, it may be of interest to install CRATE in "editable" mode by adding the -e option to the pip installation command as :code:`pip install -e .` (check `here <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_ for details).

By following this installation option, you will get the `complete CRATE project content <https://github.com/bessagroup/CRATE>`_ besides the source code, namely the documentation source and a directory with fully documented benchmarks (see :doc:`Benchmarks<../validation/benchmarks>`).

----

Using CRATE without installation
--------------------------------

It is also possible to use CRATE **without an actual installation**, provided you clone `CRATE GitHub repository <https://github.com/bessagroup/CRATE>`_ into a local directory by running the following command:

.. code-block::

   git clone git@github.com:bessagroup/CRATE.git

In this case, make sure that all the required Python package dependencies (listed in :code:`requirements.txt`) are installed. This can be done manually or automatically by running the following pip installation command:

.. code-block::

   pip install -r requirements.txt

To successfully import CRATE in your Python project, CRATE's source code directory must be then explicitly added to :code:`sys.path` as follows:

.. code-block:: python

   import sys
   # Add project directory to sys.path
   root_dir = "/path/to/project/CRATE/src"   # Replace by CRATE's source code path!
   if root_dir not in sys.path:
       sys.path.insert(0, root_dir)

   import cratepy
