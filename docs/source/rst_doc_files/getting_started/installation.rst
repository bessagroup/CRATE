
Installation
************

CRATE is a simple **Python package** (`crate <>`_) available from the Python Package Index (`PyPI <https://pypi.org/>`_). This means that running CRATE requires solely a Python 3.X installation and a few Python packages used for scientific computing and data science:

- Whether you are using Linux, MacOS or Windows, Python can be easily installed by following the `Python Getting Started <https://www.python.org/about/gettingstarted/>`_ official webpage. Many other resources are available to help you install and setup Python (e.g., `Real Python <https://realpython.com/installing-python/>`_).

- Installation **from Python Package Index**:

    - `pip <https://pip.pypa.io/en/stable/getting-started/>`_ is a Python package manager that installs Python packages from PyPI. After `installing pip <https://pip.pypa.io/en/stable/installation/>`_, installing a package is straightforward as described `here <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_ and `here <https://pip.pypa.io/en/stable/getting-started/>`_. Note that, besides installing CRATE, pip automatically installs all the required Python package dependencies. Therefore, CRATE can be simply installed by running the following pip installation command:

      .. code-block::

         pip install -U crate

    - By following this installation option, you will get `crate <>`_ newest available version on PyPI, i.e., the latest distribution version of CRATE :code:`master` branch source code uploaded to PyPI.

- Installation from **source**:

    - Clone `CRATE GitHub repository <https://github.com/bessagroup/CRATE>`_ into a local directory (check `here <https://git-scm.com/docs/git-clone>`_ for details) :

      .. code-block::

         git clone git@github.com:bessagroup/CRATE.git

    - In the cloned repository root directory, install CRATE by running the following pip installation command (check pip `regular installation <https://pip.pypa.io/en/stable/topics/local-project-installs/#regular-installs>`_ from local project):

      .. code-block::

         pip install .


      From a development point of view, it may be of interest to install CRATE in "editable" mode by adding the -e option to the pip installation command as :code:`pip install -e .` (check `here <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_ for details);

    - By following this installation option, you will get the complete CRATE project content besides the source code, namely the documentation source and a directory with fully documented benchmarks.

- It is also possible to use CRATE **without an installation**, provided you clone `CRATE GitHub repository <https://github.com/bessagroup/CRATE>`_ into a local directory. Make sure that all the required third-party package dependencies (listed in :code:`requirements.txt`) are installed - this can be done automatically by running the following pip installation command :code:`pip install -r requirements.txt`. In this case, CRATE's source code directory must be explicitly added to sys.path to successfully import crate as:

  .. code-block:: python

    import sys
    # Add project directory to sys.path
    root_dir = ‘/path/to/project/CRATE/src’   # Replace by CRATE's source code path!
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    import crate
