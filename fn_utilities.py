import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np
from scipy import linalg as la

__all__ = ['symplectic_form','exp_integrator_phi_function','trim']

""" Utility functions used accross module. """

def symplectic_form(N: int = 1) -> npt.NDArray:
    """ 2N-by-2N anti-symmetric matrix called the symplectic form """
    return np.kron(np.identity(N),np.array([[0,1],[-1,0]]))


def exp_integrator_phi_function(X: npt.NDArray, k: int = 1) -> npt.NDArray:
    """
    ϕ_k-functions used in numerical integration of linear ordinary differential equations, defined as:
        ϕ_k(X) = ∫_0^1 exp[(1-t)*X]*t^(k-1)/(k-1)! dt 
               = Σ_{l=1}^∞ X^l/(l+k)!
    The most common example is ϕ_1(X) = (exp[X]-I)/X. Solved by padding the matrix X with the identity and zeros, then
    taking the matrix exponential, and finally extracting the relevant portion to calculate ϕ_k(X). In this way, the 
    matrix inverse can be aboided when calculating ϕ_k(X), and so this algorithm will work when X is singular. For 
    time-dependent integrators, we will often calculate (exp[t*X]-I)/X = t*ϕ_1(t*X).

    ---- Parameters ----
    X : nd.array
    k : int
        Order of the ϕ_k-function, where the default is k=1. k=0 returns the usual matrix exponential.

    ---- Output ----
    ϕ_k(X) : nd.array
    """
    n = np.shape(X)[0]
    Z = np.block([np.block([[X],[np.zeros((n*k,n))]]), np.eye(n*(k+1),n*k)])
    return la.expm(Z)[0:n,n*k:n*(k+1)]


def trim(input: npt.NDArray, tol: float = qgauss.settings.auto_tidyup_atol) -> npt.NDArray:
    """ Remove small real and imaginary terms from arrays. """
    np.real(input)[np.abs(np.real(input)) < tol] = 0
    np.imag(input)[np.abs(np.imag(input)) < tol] = 0
