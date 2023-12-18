#Iterative Bragg peak Removal of X-ray Absorption Spectrum (IBR-XAS)

This package amims to provide a method of removing Bragg peaks from X-ray absorption spectra.
This method can be applied for both transmission and fluorescence mode of XAS, with the aid of varing a angle of sample.

<!-- .. image:: https://img.shields.io/pypi/v/ibr_xas.svg -->
<!-- :target: https://pypi.python.org/pypi/ibr_xas -->
<!---->
<!-- .. image:: https://img.shields.io/travis/Ameyanagi/ibr_xas.svg -->
<!-- :target: https://travis-ci.com/Ameyanagi/ibr_xas -->
<!---->
<!-- .. image:: https://readthedocs.org/projects/ibr-xas/badge/?version=latest -->
<!-- :target: https://ibr-xas.readthedocs.io/en/latest/?version=latest -->
<!-- :alt: Documentation Status -->
<!---->

## Features

-   TODO

## Limitations

This method has a limitation when there is a significant change in the trend in the post-edge region, where the assumption that following assumption is not valid anymore.
The effect of large angle rotation of the sample is expressed in the following equation, as long as the sample is uniform.

$$ \mu*0(E) \propto I*{F}/I*{0}$$
$$ \mu(E) \propto -\ln(I*{T}/I\_{0})$$

## References
