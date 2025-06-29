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


##############################################
#
#  Projections
#
##############################################

def sig_proj_convex(G, n, field, t):
    abs_G = torch.abs(G)
    G = torch.where(abs_G>=t, t/abs_G * G, G)
    G.diagonal().fill_(1.0)
    return G


def spec_proj_positive_truncated(G, m, n, field):
    lam, Q_red = torch.linalg.eigh(G)
    lam = lam[n-m:]
    Q_red = Q_red[:, n-m:]
    positive_spectrum = lam
    if field == "real":
        Gtilde = Q_red @ (positive_spectrum.unsqueeze(1) * Q_red.T)
    elif field == "complex":
        Gtilde = Q_red @ (positive_spectrum.unsqueeze(1) * Q_red.T.conj())
    return Gtilde


##############################################
#
#  Normalization
#
##############################################

def normalizeGram(G):
    xsquared = torch.diagonal(G)
    xinv = 1/torch.sqrt(xsquared)
    GHat = G * xinv.unsqueeze(1) * xinv
    return GHat

def normalizeFrame(F):
    Fhat = F @ torch.diag(1/torch.norm(F, dim=0))
    return Fhat


##############################################
#
#  Gramization
#
##############################################

def constructGram(F, field):
    if field == "real":
        G = F.T @ F
    elif field == "complex":
        G = F.T.conj() @ F
    return G


##############################################
#
#  Framization
#
##############################################

def reconstructFrame(G, m, n, field):
    positive_spectrum, Q = torch.linalg.eigh(G)
    if field == "real":
        F = torch.diag(torch.sqrt(positive_spectrum[n-m:])) @ Q[:,n-m:].T
    elif field == "complex":
        F = torch.diag(torch.sqrt(positive_spectrum[n-m:])) @ Q[:,n-m:].T.conj()
    return F


##############################################
#
#  Mutual Coherence
#
##############################################

# F must be a unit frame (F must has normalized columns)
def mutualCoherence(F, field):
    if field == "real":
        gram = F.T @ F
    elif field == "complex":
        gram = F.T.conj() @ F
    mutual_coherence = mutualCoherenceGram(gram)
    return mutual_coherence

# G must have a diagonal of ones
def mutualCoherenceGram(G):
    n = G.shape[0]
    mutual_coherence = torch.max(torch.abs(G - torch.eye(n, device=G.device)))
    return mutual_coherence


##############################################
#
#  Initialization
#
##############################################

def initializeUnitFrame(m, n, field, n_runs=1):
    if field=="real":
        F_0 = torch.randn(n_runs, m, n)
        F = torch.zeros(n_runs, m, n)
    elif field=="complex":
        F_0 = torch.randn(n_runs, m, n, dtype=torch.cfloat)
        F = torch.zeros(n_runs, m, n, dtype=torch.cfloat)
        
    for i in range(n_runs): 
        F_0[i] = normalizeFrame(F_0[i])
    return F_0, F


##############################################
#
#  Lower bound on mutual coherence
#
##############################################

import scipy.special
import numpy as np
import math

def lowerBound(m, n, field):
    if m <= 1:
        print("Can not compute coherence when m<=1.")
        return 
    if n <= 0:
        print("Can not compute coherence when n<=0.")
        return 
    
    if m>=n: #Trivial case
        return 0.0
    
    # Welch, "Lower bounds on the maximum cross correlation of signals (Corresp.)", 1974
    welch_best = 0.0
    if n>m and (field=="real" or field=="complex"): 
        degree_welch = 1
        welch_const = math.sqrt((n-m) / ((n-1)*m))
        welch_k = welch_const
        while welch_k > welch_best:
            welch_best = welch_k
            degree_welch += 1
            binom = scipy.special.comb(m+degree_welch-1, degree_welch)
            rad = (n/binom -1.0)/(n-1.0)
            welch_k = max(rad,0.0)**(1.0/(2.0*degree_welch))
    
    # Rankin, "The Closest Packing of Spherical Caps in n Dimensions", 1955
    orthoplex = 0.0
    if n>m*(m+1)/2 and field=="real":
        orthoplex = math.sqrt(1/m)
    elif n>m**2 and field=="complex":
        orthoplex = math.sqrt(1/m)
    
    # Kabatiansky and Levenshtein, "On Bounds for Packings on a Sphere and in Space", 1978
    levenshtein = 0.0
    if n>m*(m+1)/2 and field=="real":
        levenshtein = math.sqrt((3*n-m**2-2*m)/((m+2)*(n-m)))
    elif n>m**2 and field=="complex":
        levenshtein = math.sqrt((2*n-m**2-m)/((m+1)*(n-m)))
    
    # Bukh and Cox, "Nearly orthogonal vectors and small antipodal spherical codes", 2020
    buhk_cox = 0.0
    if n>m and field=="real": 
        buhk_cox = (n-m)*(n-m+1)/(2*n + (n**2 - m*n - n)*math.sqrt(2+n-m) - (n-m)*(n-m+1))
    elif n>m and field=="complex": 
        buhk_cox = (n-m)**2/(n + (n**2 - m*n - n)*math.sqrt(1+n-m) - (n-m)**2)
    
    # Xia et al., "Achieving the Welch bound with difference sets", 2005
    xia = 0.0
    if m==1 and (field=="real" or field=="complex"):
        xia=1
    elif math.log2(n)>m-1 and (field=="real" or field=="complex"):
        xia = 1 - 2*n**(-1/(m-1))
    
    # Bajwa et al., "Two are better than one: Fundamental parameters of frame coherence", 2012
    bajwa = 0.0
    if field=="real":
        coeff = 2.0**(2-m)/n* 1/scipy.special.beta(m/2, m/2)
        bajwa = max(0,math.cos(math.pi * coeff**(1/(m-1))))
    
    bounds = [welch_best, orthoplex, levenshtein, buhk_cox, xia, bajwa]
    # print(bounds)
    best_bound_index = np.argmax(bounds)
    lower_bound = bounds[best_bound_index]
    # It seems that higher order Welch bounds are always below Xia and Levenshtein bounds
    return lower_bound
