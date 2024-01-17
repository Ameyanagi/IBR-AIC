=========
Iterative Bragg Peak Removal with Automatic Intensity Correction (IBR-AIC) for X-ray Absorption Spectroscopy
=========


.. image:: https://img.shields.io/pypi/v/decomnano.svg
        :target: https://pypi.python.org/pypi/ibr-aic

.. image:: https://github.com/Ameyanagi/IBR-AIC/actions/workflows/documentation.yaml/badge.svg
        :target: https://ameyanagi.github.io/IBR-AIC/index.html
        :alt: Documentation Status

This package amims to provide a method of removing Bragg peaks from X-ray absorption spectra. This method can be applied for both transmission and fluorescence mode of XAS, with the aid of varing a angle of sample.

* Free software: MIT license
* Documentation: https://ameyanagi.github.io/IBR-AIC/index.html.

Requirements
------------

IBR-AIC relies on numpy_ and scipy_ package as dependencies. Please install numpy_ and scipy_.\
The requirements can be optionally installed through requirements.txt by following command.

Group from xraylarch_ package can also be used for the input of the calculation, if so,  please install xraylarch_ through pip.

.. _numpy: https://numpy.org/
.. _scipy: https://scipy.org/
.. _xraylarch: https://xraypy.github.io/xraylarch/

.. code-block:: bash

    pip install -r requirements.txt

Installation
------------

Installation currently done through following command.

.. code-block:: bash

   pip install git+https://github.com/Ameyanagi/IBR-AIC

.. Detailed instructions for installation are available in the `installation documentation`_.

.. _installation documentation: https://ameyanagi.github.io/DecomNano/installation.html

Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~~

(This method is not available yet. It will be available after the submission.)

.. code-block:: bash

    pip install ibr-aic


Usage
-----

Python API
~~~~~~~~~~

Detailed instructions for usage are available in the `API documentation`_.
The decomanano package can be imported and used in python scripts.

.. _API documentation: https://ameyanagi.github.io/IBR-AIC/modules.html

Limitation
----------

This method has a limitation when there is a significant change in the trend in the post-edge region, where the assumption that following assumption is not valid anymore. The effect of large angle rotation of the sample is expressed in the following equation, as long as the sample is uniform.

.. math::
    \mu(E) &\propto I_F/I_0

    \mu(E) &\propto -\ln(I_T/I_0)

Citation
--------

If you use IBR-AIC in your research, please cite the following paper: to be submitted.
