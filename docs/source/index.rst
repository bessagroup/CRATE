
Welcome to Sphinx fellow member of Bessa Research Group!
========================================================

Where does this content come from?
----------------------------------
This is the content of your ``index.rst`` file.


reStructuredText language (``.rst``)
------------------------------------
The ``.rst`` extension stands for `reStructuredText language <https://docutils.sourceforge.io/rst.html>`_, an easy-to-read, what-you-see-is-what-you-get plaintext markup syntax and parser system. It is useful for in-line program documentation (such as Python docstrings), for quickly creating simple web pages, and for standalone documents. It is also the default plaintext markup language used by both Docutils and Sphinx.

Everything you need to learn `reStructuredText language <https://docutils.sourceforge.io/rst.html>`_ can be found `here <https://docutils.sourceforge.io/rst.html>`_. If you want to dive right into the syntax, then this `Cheat Sheet <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_ will come in handy as well as `Basic Guide <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. There are several open-source tools that support reStructuredText and where you can experiment your syntax (e.g., `Online reStructure Editor <https://www.tutorialspoint.com/online_restructure_editor.php>`_).

----

Examples of Roles
=================

Cross-referencing syntax
------------------------

.. rubric:: Cross-referencing arbitrary locations

.. rubric:: \:doc:

A cross-reference to Document 1: :doc:`documentation/doc_page_1`

A cross-reference to Document 1 (customized link text): :doc:`here <documentation/doc_page_1>`

.. rubric:: \:ref:


Cross-referencing Python objects
--------------------------------




Inline code highlighting
------------------------

.. rubric:: \:code:

An example of a :code:`inline code` (literal).

.. rubric:: \:python:

.. role:: python(code)
   :language: python

An example of a Python custom role inline code: :python:`bool = False or 1 > 2 or 'abc'[0] == 'd'`


Math
----

.. rubric:: \:math:

An example of a inline math formula: :math:`a^2 + b^2 = c^2`


Other semantic markup
---------------------

.. rubric:: \:abbr:

An example of an abbreviation: :abbr:`PyPI (Python Package Index)`

.. rubric:: \:command:

An example of a OS-level command: :command:`cd`


Substitutions
-------------

Get today's date automatically: |today|

Get configuration file version automatically: |version|

Get configuration file release automatically: |release|


----

Examples of Directives
======================

Table of contents
-----------------

.. rubric:: .. toctree

.. toctree::
   :name: mastertoc
   :caption: Caption of example toctree 1
   :maxdepth: 3
   :numbered:
   :includehidden:

   documentation/doc_page_1
   documentation/doc_page_2
   documentation/doc_folder_1/doc_page_3

.. toctree::
   :name: othertoc
   :caption: Caption of example toctree 2
   :maxdepth: 3
   :includehidden:

   documentation/doc_page_1
   documentation/doc_page_2
   documentation/doc_folder_1/doc_page_3



Paragraph-level markup
----------------------

.. rubric:: .. note

.. note::

   This is a note.

.. rubric:: .. warning

.. warning::

   This is a warning.

.. rubric:: .. versionadded

.. versionadded:: X.Y

   This indicates the version of the project which added the described feature.

.. rubric:: .. versionchanged

.. versionchanged:: X.Y

   This indicates the version of the project which changed the described feature.

.. rubric:: .. seealso

.. seealso::

   :py:mod:`zipfile`: Documentation of the :py:mod:`zipfile` standard module.

.. rubric:: .. hlist:

.. hlist::
   :columns: 3

   * A list of
   * short items
   * that should be
   * displayed
   * horizontally
   * like this.


Code examples
-------------

.. rubric:: .. code-block

.. code-block:: python

   # This is a Python code block
   x = 1

.. code-block:: python
   :linenos:
   :lineno-start: 3
   :emphasize-lines: 3, 4
   :caption: Caption of the code block may go here
   :name: example_code_block


   # This is a Python code block with some customization
   x = 1
   # Including emphasized lines
   y = 2


Meta-information
----------------

.. rubric:: .. sectionauthor

.. sectionauthor:: section_author_name <author@email>


Index-generating markup
-----------------------

.. rubric:: .. index

.. index::
   single: A custom single index

You can find the single index entry 'A custom single index' on the :ref:`genindex`.


Math
----

.. rubric:: .. math

.. math::
   :nowrap:

   A simple LaTeX mathematical equation

   \begin{equation}
      (a + b)^2 = a^2 + 2ab + b^2
   \end{equation}


Glossary
--------

.. rubric:: .. glossary

.. glossary::

   This is a term
      Description of the term.

   This is another term
      Definition of the term.

----

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

----


.. toctree::
   :name: apitoc
   :caption: API
   :hidden:

   API reference <_autosummary/crate>
