"""
Grassmannian Frame Computation via Accelerated Alternating Projections

30 January 2025 - v1.0

Bastien MASSION
UCLouvain, ICTEAM
bastien.massion@uclouvain.be

Prof. Estelle MASSART
UCLouvain, ICTEAM
estelle.massart@uclouvain.be

Adapted to torch by Binxu Wang
Harvard, Kempner Institute
binxu_wang@hms.harvard.edu
28 June 2025
"""

import numpy as np
import time

import torch as th
from taap_utils_torch import initializeUnitFrame, mutualCoherence, lowerBound
from taap_torch import taap


##############################################
#
#  Dimensions and field
#
##############################################

m = 768
# n = 4096
n = 32000
field = "real"
# field = "complex"
print("Dimensions: \t(%d, %d)" %(m,n))
print("Field: \t \t \t" + field)
lb = lowerBound(m, n, field)
print("Lower bound: \t%.6f" %lb)


##############################################
#
#  TAAP hyperparameters
#
##############################################

beta    = 2.0
N_budg  = 2000
tau     = 10**(-6)
N_p     = 50
eps_p   = 10**(-3)
eps_s   = 10**(-1)
accel   = True
verbose = True
device = "cuda"
print_every = 10

##############################################
#
#  Run TAAP
#
##############################################

n_runs = 1
F_0, F_run = initializeUnitFrame(m, n, field, n_runs=n_runs)
mu_run = th.zeros(n_runs)
N_tot_run = th.zeros(n_runs)
duration_run = th.zeros(n_runs)
time_per_it_run = th.zeros(n_runs)

for run_index in range(n_runs):
    mu_0 = mutualCoherence(F_0[run_index], field)
    
    start_time = time.time()
    F_run[run_index], mu_run[run_index], N_tot_run[run_index] = taap(F_0[run_index], m, n, field, beta=beta, N_budg=N_budg, tau=tau, N_p=N_p, eps_p=eps_p, eps_s=eps_s, acceleration=accel, verbose=verbose, print_every=print_every, device=device)
    end_time = time.time()
    duration_run[run_index] = end_time-start_time
    time_per_it_run[run_index] = duration_run[run_index]/N_tot_run[run_index]
    print("Runtime for run %d: \t%.3f" %(run_index+1,duration_run[run_index]))
    print("Runtime per iteration: \t%.6f" %time_per_it_run[run_index])
    print("Final mutual coherence: %.6f" %mu_run[run_index])
    # print(F_run[run_index])
    print()

save_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/Grassmannian"
th.save(F_run, save_path + "/F_run_768_32000.pt")
th.save({"duration_run":duration_run, "time_per_it_run":time_per_it_run, "mu_run":mu_run, "N_tot_run":N_tot_run}, 
        save_path + "/taap_results_768_32000.pt")


