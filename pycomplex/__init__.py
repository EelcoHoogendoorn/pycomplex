"""
Discrete exterior calculus in python

This is an attempt to unify code that i have written over the years in my own projects, and
also draws inspiration from pyDEC

It aims to provide all required operations in a fully vectorized manner; that is,
with decent efficiency useful for many real world tasks

Also, it tries to be very general; providing support for both simplicial and regular topologies,
and different embeddings and metrics, while maximizing code and conceptual reuse between these variants

"""