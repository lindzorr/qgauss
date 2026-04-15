import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np
import scipy

from .qgstate import *
from .qgoper import *
from .qgsuper import *

__all__ = ['vacuum','thermal','displaced','sm_squeeze','tm_squeeze',
           'qubit_excited','qubit_ground',
           'destroy','create','position','momentum',
           'one','identity','identity_cvs','num',
           'qeye','identity_fls','qzero','basis',
           'sigmam','sigmap','sigmax','sigmay','sigmaz','jmat'
          ]

""" Basic constructors to create often used states and operators. """

'''
------------
   States   
------------
'''

def vacuum(N = 1) -> QGstate:
    # N-mode vacuum state
    return QGstate(data_2nd = 0.5*np.identity(2*N), 
                   dims_cvs = N)

def thermal(*nth: float | list[float] | npt.NDArray[float]) -> QGstate:
    # N-mode thermal state, with occupancies "nth"
    if not nth:
        nth = 0
    elif len(nth) == 1 and isinstance(nth[0], (numbers.Number, np.number)):
        nth = np.asarray([nth[0]])
    elif len(nth) == 1 and isinstance(nth[0], list | np.ndarray):
        nth = np.asarray(nth[0])
    return QGstate(data_2nd = np.kron(np.diag(nth)+0.5*np.identity(len(nth)),np.identity(2)), 
                   dims_cvs = len(nth))

def displaced(alpha: complex | list[float] | npt.NDArray[float] = None) -> QGstate:
    # Single mode coherent/displaced vacuum state
    if alpha is None:
        q, p = 0, 0
    elif isinstance(alpha, (numbers.Number, np.number)):
        q, p = np.real(alpha), np.imag(alpha)
    elif isinstance(alpha, np.ndarray | list) and len(alpha) == 2:
        q, p = alpha[0], alpha[1]
    return QGstate(data_2nd = 0.5*np.identity(2),
                   data_1st = np.array([q,p]),
                   dims_cvs = 1)

def sm_squeeze(sqz: complex | list[float] | npt.NDArray[float] = None) -> QGstate:
    # Single-mode squeezed state
    if sqz is None:
        r, t = 0, 0
    elif isinstance(sqz, (numbers.Number, np.number)):
        r, t = np.abs(sqz), np.angle(sqz)
    elif isinstance(sqz, np.ndarray | list) and len(sqz) == 2:
        r, t = sqz[0], sqz[1]
    return QGstate(data_2nd = 0.5*np.array([[np.cosh(2*r) + np.cos(t)*np.sinh(2*r), -np.sin(t)*np.sinh(2*r)],
                                            [-np.sin(t)*np.sinh(2*r), np.cosh(2*r) - np.cos(t)*np.sinh(2*r)]]),
                   dims_cvs = 1)

def tm_squeeze(sqz: complex | list[float] | npt.NDArray[float] = None) -> QGstate:
    # Two-mode squeezed state
    if sqz is None:
        r, t = 0, 0
    elif isinstance(sqz, (numbers.Number, np.number)):
        r, t = np.abs(sqz), np.angle(sqz)
    elif isinstance(sqz, np.ndarray | list) and len(sqz) == 2:
        r, t = sqz[0], sqz[1]
    return QGstate(data_2nd = 0.5*np.array([[np.cosh(2*r), 0, -np.cos(t)*np.sinh(2*r), -np.sin(t)*np.sinh(2*r)],
                                            [0, np.cosh(2*r), -np.sin(t)*np.sinh(2*r), +np.cos(t)*np.sinh(2*r)],
                                            [-np.cos(t)*np.sinh(2*r), -np.sin(t)*np.sinh(2*r), np.cosh(2*r), 0],
                                            [-np.sin(t)*np.sinh(2*r), +np.cos(t)*np.sinh(2*r), 0, np.cosh(2*r)]]),
                   dims_cvs = 2)

def qubit_excited() -> QGstate:
    # Density matrix for excited qubit state
    return QGstate(data_0th = np.array([[1,0],[0,0]]),
                   dims_fls = [[2],[2]])

def qubit_ground() -> QGstate:
    # Density matrix for ground qubit state
    return QGstate(data_0th = np.array([[0,0],[0,1]]),
                   dims_fls = [[2],[2]])

'''
---------------
   Operators   
---------------
'''

def one() -> QGoper:
    # Simple "identity" operator for single continuous-variable system
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = 1)

def identity(N: int = 1) -> QGoper:
    # Identity operator for N-mode continuous-variable system
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = N)

def identity_cvs(N: int = 1) -> QGoper:
    # Alias of identity_cvs
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = N)

def destroy() -> QGoper:
    # Single mode annihilation operator
    return QGoper(data_1st = np.array([1,1j])/np.sqrt(2), 
                  dims_cvs = 1)

def create() -> QGoper:
    # Single mode creation operator
    return QGoper(data_1st = np.array([1,-1j])/np.sqrt(2), 
                  dims_cvs = 1)

def position() -> QGoper:
    # Single mode position operator
    return QGoper(data_1st = np.array([1,0]), 
                  dims_cvs = 1)
    
def momentum() -> QGoper:
    # Single mode momentum operator
    return QGoper(data_1st = np.array([0,1]), 
                  dims_cvs = 1)

def num() -> QGoper:
    # Single mode number operator
    return QGoper(data_2nd = np.array([[1,0],[0,1]]), 
                  data_0th = -1/2, 
                  dims_cvs = 1)

def identity_fls(dims: int | list[int] | npt.NDArray[int] = 1) -> QGoper:
    # N-by-N identity operator for a finite-level system, where N may an integer or list of integers
    if isinstance(dims, (np.ndarray, list)):
        return QGoper(data_0th = np.identity(np.sum(dims)), 
                      dims_fls = [dims,dims])
    else:
        return QGoper(data_0th = np.identity(dims), 
                      dims_fls = [[dims],[dims]])

def qeye(N: int = 1) -> QGoper:
    # N-by-N identity operator for a finite-level system
    return QGoper(data_0th = np.identity(N), 
                  dims_fls = [[N],[N]])

def qzero(N: int = 1) -> QGoper:
    # N-by-N null operator for a finite-level system
    return QGoper(data_0th = np.zeros(N), 
                  dims_fls = [[N],[N]])

def basis(m: int, N: int = 1) -> QGoper:
    # N-by-N operator corresponding to the outer product of a single basis vector, |m><m|.
    # To obey the convention used here for FLs operators, |0><0| has a one in the bottom-right corner,
    # while |N-1><N-1| is in the upper-left corner of the data matrix.
    data = np.zeros((N, N))
    data[N-m-1, N-m-1] = 1
    return QGoper(data_0th = data, 
                  dims_fls = [[N],[N]])

def sigmam() -> QGoper:
    # Lowering operator for a TLS/qubit
    return QGoper(data_0th = np.array([[0,0],[1,0]]), 
                  dims_fls = [[2],[2]])

def sigmap() -> QGoper:
    # Raising operator for a TLS/qubit
    return QGoper(data_0th = np.array([[0,1],[0,0]]), 
                  dims_fls = [[2],[2]])

def sigmax() -> QGoper:
    # Pauli x-operator
    return QGoper(data_0th = np.array([[0,1],[1,0]]), 
                  dims_fls = [[2],[2]])

def sigmay() -> QGoper:
    # Pauli y-operator
    return QGoper(data_0th = np.array([[0,-1j],[1j,0]]), 
                  dims_fls = [[2],[2]])

def sigmaz() -> QGoper:
    # Pauli z-operator
    return QGoper(data_0th = np.array([[1,0],[0,-1]]), 
                  dims_fls = [[2],[2]])

def jmat(j, comp: str) -> QGoper:
    # Spin operator with total spin "j", where the string "comp" denotes the component of the spin,
    # and only takes values ['+','-','x','y','z','tot'].
    if j < 0:
        raise TypeError("The total spin j must be a non-negative integer.")
    if 2*j != np.fix(2*j):
        raise TypeError("The total spin j can only take integer or half-integer values.")

    if comp == '+':
        data = np.diag([np.sqrt(j*(j+1)-m*(m+1)) for m in np.arange(j-1,-j-1,-1)], k=1)
    elif comp == '-':
        data = np.diag([np.sqrt(j*(j+1)-m*(m-1)) for m in np.arange(j,-j,-1)], k=-1)
    elif comp == 'x':
        data = 0.5*(np.diag([np.sqrt(j*(j+1)-m*(m+1)) for m in np.arange(j-1,-j-1,-1)], k=1)
                    + np.diag([np.sqrt(j*(j+1)-m*(m-1)) for m in np.arange(j,-j,-1)], k=-1))
    elif comp == 'y':
        data = -0.5j*(np.diag([np.sqrt(j*(j+1)-m*(m+1)) for m in np.arange(j-1,-j-1,-1)], k=1)
                      - np.diag([np.sqrt(j*(j+1)-m*(m-1)) for m in np.arange(j,-j,-1)], k=-1))
    elif comp == 'z':
        data = np.diag([m for m in np.arange(j,-j-1,-1)])
    elif comp == 'tot':
        data = j*(j+1)*np.identity(int(2*j+1))
    else:
        raise TypeError("A valid component of the spin must be provided. Choose one of ['+','-','x','y','z','tot'].")
                        
    return QGoper(data_0th = data,
                  dims_fls = [[int(2*j+1)],[int(2*j+1)]])