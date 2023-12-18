=======
Iterative Bragg peak Removal of X-ray Absorption Spectrum (IBR-XAS)
=======

This package amims to provide a method of removing Bragg peaks from X-ray absorption spectra.
This method can be applied for both transmission and fluorescence mode of XAS, with the aid of varing a angle of sample.



.. image:: https://img.shields.io/pypi/v/ibr_xas.svg
        :target: https://pypi.python.org/pypi/ibr_xas

.. image:: https://img.shields.io/travis/Ameyanagi/ibr_xas.svg
        :target: https://travis-ci.com/Ameyanagi/ibr_xas

.. image:: https://readthedocs.org/projects/ibr-xas/badge/?version=latest
        :target: https://ibr-xas.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Iterative Bragg Peak removal from X-ray absoption spectra


* Free software: MIT license
* Documentation: https://ibr-xas.readthedocs.io.


Features
--------

* TODO


Limitations
-----------

This method has a limitation when there is a significant change in the trend in the post-edge region, where the assumption that

    .. math::
        \mu_0(E) \propto I_{F}/I_{0}
        \mu(E) \propto -\ln(I_{T}/I_{0})

References
-------

