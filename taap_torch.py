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

import torch
import math
import numpy as np
from taap_utils_torch import sig_proj_convex, spec_proj_positive_truncated, normalizeGram, constructGram, reconstructFrame, mutualCoherence, mutualCoherenceGram, lowerBound


##############################################
#
#  TAAP
#
##############################################


def taap(F_0, m, n, field, 
         beta=2.0, 
         N_budg=100000, 
         tau=10**(-6), 
         N_p=100, 
         eps_p=10**(-1), 
         eps_s=10**(-3), 
         acceleration=True, 
         verbose=True, 
         print_every=10,
         device="cpu"):
    
    if verbose == True:
        print("mu_0_AAP \tmu_AAP \t\ttarget \t\tdelta_t \tN_AAP \tN_tot")
    
    N_tot = 0
    F_0 = F_0.to(device)
    G_best = constructGram(F_0, field)
    mu_best = mutualCoherence(F_0, field)
    theoretical_lower_bound = lowerBound(m, n, field)
    
    t = theoretical_lower_bound
    delta_t = mu_best - t
    try:
        while not (delta_t < tau or N_tot > N_budg):
            G_AAP = G_best
            mu_AAP = mu_best
            k_AAP = 0
            
            c_k_1 = 1.0
            G_k_2 = G_best
            G_k_1 = G_best
            k = 1
            
            while not (mu_AAP - t < eps_s*delta_t or k - k_AAP > N_p):
                if acceleration == True: 
                    c_k = math.sqrt(4*c_k_1**2 + 1)/2 + 1/2
                    Y_k = G_k_1 + (c_k_1 - 1.0)/c_k * (G_k_1-G_k_2)
                    G_k = spec_proj_positive_truncated(sig_proj_convex(Y_k, n, field, t), m, n, field)
                
                else:
                    G_k = spec_proj_positive_truncated(sig_proj_convex(G_k_1, n, field, t), m, n, field)
                
                mu_k = mutualCoherenceGram(normalizeGram(G_k))
                
                if mu_AAP - mu_k > eps_p * delta_t:
                    G_AAP = G_k
                    mu_AAP = mu_k
                    k_AAP = k
                
                k += 1
                G_k_2 = G_k_1
                G_k_1 = G_k
                c_k_1 = c_k
                N_tot += 1

                if verbose == True and k % print_every == 0:
                    print("%.6f \t%.6f \t%.6f \t%.6f \t%-6d \t%-6d" %(mu_best, mu_AAP, t, delta_t, k-1, N_tot))
                
            if verbose == True:
                print("%.6f \t%.6f \t%.6f \t%.6f \t%-6d \t%-6d" %(mu_best, mu_AAP, t, delta_t, k-1, N_tot))
            
            if mu_AAP - t < eps_s*delta_t:
                t = torch.max(torch.tensor([mu_AAP - beta*delta_t, theoretical_lower_bound]))
            
            elif k - k_AAP > N_p:
                t = torch.max(torch.tensor([mu_AAP - 1/beta*delta_t, theoretical_lower_bound]))
            
            G_best = normalizeGram(G_AAP)
            mu_best = mu_AAP
            delta_t = mu_best - t
    
    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user (Ctrl+C).")
            print(f"Best so far: μ={mu_best:.6f}, iterations={N_tot}")
        # reconstruct and immediately return the best frame so far
        F_best = reconstructFrame(G_best, m, n, field).cpu()
        return F_best, mu_best, N_tot

    F_best = reconstructFrame(G_best, m, n, field).cpu()
    return F_best, mu_best, N_tot


import time
import torch as th
from taap_utils_torch import initializeUnitFrame, mutualCoherence, lowerBound

def run_taap_torch(m=16, n=128, 
                   field="real", # "real" or "complex"
                   n_runs = 1,
                   beta=2.0, 
                   N_budg=100000, 
                   tau=10**(-6), 
                   N_p=100, 
                   eps_p=10**(-3), 
                   eps_s=10**(-1), 
                   accel=True, 
                   verbose=True, 
                   print_every=10,
                   device="cuda"):

    print("Dimensions: \t(%d, %d)" %(m,n))
    print("Field: \t \t \t" + field)
    lb = lowerBound(m, n, field)
    print("Lower bound: \t%.6f" %lb)

    ##############################################
    #
    #  Run TAAP
    #
    ##############################################

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
    return F_run, mu_run, N_tot_run, duration_run, time_per_it_run


