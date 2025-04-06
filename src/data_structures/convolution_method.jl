"""
    ConvolutionMethod

A type representing the convolution-based interpolation method.

This is used as a flag type to dispatch to the appropriate interpolation algorithms
in the interpolation type system. It represents a strategy for interpolation based on
convolution with specialized kernels.

The `ConvolutionMethod` is typically used internally within the interpolation objects
and is not generally manipulated directly by users.
"""
struct ConvolutionMethod end