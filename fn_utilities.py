import numpy.typing as npt
import qgauss
import numpy as np
from scipy import linalg as la

__all__ = ['symplectic_form','exp_integrator_phi_function','trim',
           'mat_to_vec','vec_to_mat','symmat_to_vec','vec_to_symmat']

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


def trim(input: npt.NDArray, tol: float = qgauss.settings.tidyup_atol) -> npt.NDArray:
    """ Remove small real and imaginary terms from arrays. """
    np.real(input)[np.abs(np.real(input)) < tol] = 0
    np.imag(input)[np.abs(np.imag(input)) < tol] = 0


def mat_to_vec(input: npt.NDArray, order = 'C') -> npt.NDArray:
    """ Vectorization of a matrix, select 'C' for column stacking and 'R' for row stacking. """
    if order == 'C':
        return input.ravel(order = 'C')
    elif order == 'R':
        return input.ravel(order = 'R')
    else:
        raise TypeError("Order of vectorization must be column-stacking (C) or row-stacking (R).")
    

def vec_to_mat(input: npt.NDArray, shape: tuple[int], order = 'C') -> npt.NDArray:
    """ Inverse of the vectorization of a matrix, select 'C' for column stacking and 'R' for row stacking. """
    if order == 'C':
        return np.reshape(input, shape, order = 'C')
    elif order == 'R':
        return np.reshape(input, shape, order = 'R')
    else:
        raise TypeError("Order of inverse-vectorization must be convert vector into matrix columns (C) or rows (R).")


def symmat_to_vec(input: npt.NDArray) -> npt.NDArray:
    """ Vectorization where a NxN symmetric matrix is converted to a vector columnwise,
    saving only the unique elements, resulting in a vector of length N(N+1)/2. """
    return (input)[np.tril_indices(input.shape[0])]


def vec_to_symmat(input: npt.NDArray, dims: int) -> npt.NDArray:
    """ Convert a vector of length N(N+1)/2 into a NxN symmetric matrix. """
    mask = np.tri(dims, dtype = bool , k = 0)
    out = np.zeros((dims,dims), dtype = complex)
    out[mask] = input
    return out + np.triu(np.transpose(out),1)