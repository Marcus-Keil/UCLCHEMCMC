import numpy as np
from spectralradex import radex
from pandas import DataFrame, concat
from multiprocessing import pool
from functools import partial

import time

# Single run using just the basic run() method from Spectral Radex

params = radex.get_default_parameters()
params["molfile"] = "co.dat"
output = radex.run(params)


# Usage example of run_grid() method and how to use or not use multiprocessing with it.

tic = time.perf_counter()
grid_DF = radex.run_grid(density_values=np.arange(1.0e5, 1.0e6, 1.0e5), temperature_values=np.arange(10, 100, 10),
                   column_density_values=np.arange(1.0e14, 1.0e15, 1.0e14), molfile='co.dat',
                   target_value="T_R (K)", pool=None)
toc = time.perf_counter()
print(f"run_grid took {toc-tic:0.4f} seconds without a pool")

tic = time.perf_counter()
grid_DF_pool = radex.run_grid(density_values=np.arange(1.0e5, 1.0e6, 1.0e5), temperature_values=np.arange(10, 100, 10),
                   column_density_values=np.arange(1.0e14, 1.0e15, 1.0e14), molfile='co.dat',
                   target_value="T_R (K)", pool=pool.Pool())
toc = time.perf_counter()
print(f"run_grid took {toc-tic:0.4f} seconds with a pool of 4 workers")
print(grid_DF_pool.head())