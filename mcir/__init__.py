# -*- coding: utf-8 -*-

"""Top-level package for Multi-component Inversion Recovery Analysis of Cortical Layers."""
__author__ = """Omri Tomer"""
__email__ = 'omritomer1@mail.tau.ac.il'
__version__ = '0.0.1'

import sys
if (sys.version_info < (3, 3)):
    from t1_calc import T1Calc
else:
    from mcir.t1_calc import T1Calc
