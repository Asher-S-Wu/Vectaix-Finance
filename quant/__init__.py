"""Vectaix 本地量化研究。"""

import os

# Limit numerical libraries before any entry point imports NumPy.
for _setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_setting] = '1'
