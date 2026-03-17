import types
import warnings
import numbers
import numpy as np

from .qgstate import *
from .qgoper import *
from .qgsuper import *

__all__ = ['symplectic_form',
           'vacuum','thermal','displaced','sm_squeeze','tm_squeeze',
           'qubit_excited','qubit_ground',
           'destroy','create','position','momentum',
           'one','identity','identity_cvs','num',
           'qeye','identity_fls','qzero',
           'sigmam','sigmap','sigmax','sigmay','sigmaz',
           'spre','spost','sprepost','anticommutator',
           'dissipator','coherent','lindbladian','symplectic_form'
          ]

"""
This is the current home for constructors used to create often used states, operators, and superoperators.
"""

def symplectic_form(N=1):
    # 2N-by-2N anti-symmetric matrix called the symplectic form
    return np.kron(np.identity(N),np.array([[0,1],[-1,0]]))

# Constructors for states

def vacuum(N=1):
    # N-mode vacuum state
    return QGstate(data_2nd = 0.5*np.identity(2*N), 
                   dims_cvs = N)

def thermal(*nth):
    # N-mode thermal state, with occupancies "nth"
    if not nth:
        nth = 0
    elif len(nth) == 1 and isinstance(nth[0], (numbers.Number, np.number)):
        nth = np.asarray([nth[0]])
    elif len(nth) == 1 and isinstance(nth[0], list | np.ndarray):
        nth = np.asarray(nth[0])
    return QGstate(data_2nd = np.kron(np.diag(nth)+0.5*np.identity(len(nth)),np.identity(2)), 
                   dims_cvs = len(nth))

def displaced(alpha=None):
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

def sm_squeeze(sqz=None):
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

def tm_squeeze(sqz=None):
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

def qubit_excited():
    # Density matrix for excited qubit state
    return QGstate(data_0th = np.array([[1,0],[0,0]]),
                   dims_fls = [[2],[2]])

def qubit_ground():
    # Density matrix for ground qubit state
    return QGstate(data_0th = np.array([[0,0],[0,1]]),
                   dims_fls = [[2],[2]])

# Constructors for operators

def one():
    # Simple "identity" operator for single continuous-variable system
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = 1)

def identity(N=1):
    # Identity operator for N-mode continuous-variable system
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = N)

def identity_cvs(N=1):
    # Alias of identity_cvs
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = N)

def destroy():
    # Single mode annihilation operator
    return QGoper(data_1st = np.array([1,1j])/np.sqrt(2), 
                  dims_cvs = 1)

def create():
    # Single mode creation operator
    return QGoper(data_1st = np.array([1,-1j])/np.sqrt(2), 
                  dims_cvs = 1)

def position():
    # Single mode position operator
    return QGoper(data_1st = np.array([1,0]), 
                  dims_cvs = 1)
    
def momentum():
    # Single mode momentum operator
    return QGoper(data_1st = np.array([0,1]), 
                  dims_cvs = 1)

def num():
    # Single mode number operator
    return QGoper(data_2nd = np.array([[1,0],[0,1]]), 
                  data_0th = -1/2, 
                  dims_cvs = 1)

def identity_fls(dims=1):
    # N-by-N identity operator for a finite-level system, where N may an integer or list of integers
    if isinstance(dims, (np.ndarray, list)):
        return QGoper(data_0th = np.identity(np.sum(dims)), 
                      dims_fls = [dims,dims])
    else:
        return QGoper(data_0th = np.identity(dims), 
                      dims_fls = [[dims],[dims]])


def qeye(N=1):
    # N-by-N identity operator for a finite-level system
    return QGoper(data_0th = np.identity(N), 
                  dims_fls = [[N],[N]])

def qzero(N=1):
    # N-by-N null operator for a finite-level system
    return QGoper(data_0th = np.zeros(N), 
                  dims_fls = [[N],[N]])
    
def sigmam():
    # Lowering operator for a TLS/qubit
    return QGoper(data_0th = np.array([[0,0],[1,0]]), 
                  dims_fls = [[2],[2]])

def sigmap():
    # Raising operator for a TLS/qubit
    return QGoper(data_0th = np.array([[0,1],[0,0]]), 
                  dims_fls = [[2],[2]])

def sigmax():
    # Pauli x-operator
    return QGoper(data_0th = np.array([[0,1],[1,0]]), 
                  dims_fls = [[2],[2]])

def sigmay():
    # Pauli y-operator
    return QGoper(data_0th = np.array([[0,-1j],[1j,0]]), 
                  dims_fls = [[2],[2]])

def sigmaz():
    # Pauli z-operator
    return QGoper(data_0th = np.array([[1,0],[0,-1]]), 
                  dims_fls = [[2],[2]])


# Constructors for superoperators

def dissipator(a, b=None):
    if b is None:
        b = a
    return sprepost(a,b.dag()) - 0.5*spre(b.dag()*a) - 0.5*spost(b.dag()*a)

def anticommutator(H):
    return (spre(H) + spost(H))

def coherent(H):
    return -1.0j*(spre(H) - spost(H))

def lindbladian(H=None, c_ops=[]):
    if H is not None:
        L = -1.0j*(spre(H) - spost(H))
    else:
        c0 = c_ops.pop(0)
        L = sprepost(c0,c0.dag()) - 0.5 * spre(c0.dag()*c0) - 0.5 * spost(c0.dag()*c0)
    L += sum([sprepost(c,c.dag()) - 0.5 * spre(c.dag()*c) - 0.5 * spost(c.dag()*c) for c in c_ops])
    return L

def spost(A):
    # Superoperator representing post/right-multiplication of state by an operator
    if not isinstance(A, QGoper):
        raise TypeError("Input is not of type: QGoper")
    
    if A.isfls:
        return QGsuper(data_2nd_r = np.einsum('jpkqyz,plqm->kljmyz',
                                                A.data_2nd[:,np.newaxis,:,np.newaxis,:,:],
                                                np.identity(np.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                               ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                         np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                         2*A.dims_cvs,
                                                         2*A.dims_cvs),
                       data_1st_r = np.einsum('jpkqz,plqm->kljmz',
                                              A.data_1st[:,np.newaxis,:,np.newaxis,:],
                                              np.identity(np.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                             ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                       np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                       2*A.dims_cvs),
                       data_0th = np.einsum('jpkq,plqm->kljm',
                                            A.data_0th[:,np.newaxis,:,np.newaxis],
                                            np.identity(np.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                           ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                     np.prod((A.dims_fls[0], A.dims_fls[1]))),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], A.dims_fls[1]],
                                   [A.dims_fls[0], A.dims_fls[1]]])
    
    else:
        return QGsuper(data_2nd_r = A.data_2nd,
                       data_1st_r = A.data_1st,
                       data_0th = A.data_0th,
                       dims_cvs = A.dims_cvs)

def spre(A):
    # Superoperator representing pre/left-multiplication of state by an operator
    if not isinstance(A, QGoper):
        raise TypeError("Input is not of type: QGoper")
    
    if A.isfls:
        return QGsuper(data_2nd_l = np.einsum('jpkq,plqmyz->jlkmyz',
                                              np.identity(np.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                              A.data_2nd[np.newaxis,:,np.newaxis,:,:,:]
                                             ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                       np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                       2*A.dims_cvs,
                                                       2*A.dims_cvs),
                       data_1st_l = np.einsum('jpkq,plqmz->jlkmz',
                                              np.identity(np.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                              A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                             ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                       np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                       2*A.dims_cvs),
                       data_0th = np.einsum('jpkq,plqm->jlkm',
                                            np.identity(np.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                            A.data_0th[np.newaxis,:,np.newaxis,:]
                                           ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                     np.prod((A.dims_fls[0], A.dims_fls[1]))),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], A.dims_fls[1]],
                                   [A.dims_fls[0], A.dims_fls[1]]])
                                     
    else:
        return QGsuper(data_2nd_l = A.data_2nd,
                       data_1st_l = A.data_1st,
                       data_0th = A.data_0th,
                       dims_cvs = A.dims_cvs)

def sprepost(A, B):
    # Superoperator representing pre/left and post-right-multiplication of state by an operator
    if not (isinstance(A, QGoper) or isinstance(B, QGoper)):
        raise TypeError("Input is not of type: QGoper")

    if (A.dims_cvs != B.dims_cvs) and (A.dims_fls != B.dims_fls).all():
        raise ValueError("Inputs do not have identical dimensions")

    if ((A.is2nd and B.is2nd) or
        (A.is2nd and B.is1st) or
        (A.is1st and B.is2nd)
        ):
        raise ValueError("Inputs result in superoperator which is not Gaussian")

    if A.isfls and B.isfls:
        return QGsuper(data_2nd_l = np.einsum('jpkq,plqmyz->kljmyz',
                                              B.data_0th[:,np.newaxis,:,np.newaxis],
                                              A.data_2nd[np.newaxis,:,np.newaxis,:,:,:]
                                             ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                       np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                       2*A.dims_cvs,
                                                       2*A.dims_cvs),
                       data_2nd_r = np.einsum('jpkqyz,plqm->kljmyz',
                                              B.data_2nd[:,np.newaxis,:,np.newaxis,:,:],
                                              A.data_0th[np.newaxis,:,np.newaxis,:]
                                             ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                       np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                       2*B.dims_cvs,
                                                       2*B.dims_cvs),
                       data_2nd_m = np.einsum('jpkqz,plqmy->kljmyz',
                                              B.data_1st[:,np.newaxis,:,np.newaxis,:],
                                              A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                             ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                       np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                       2*A.dims_cvs,
                                                       2*B.dims_cvs),
                       data_1st_l = np.einsum('jpkq,plqmz->kljmz',
                                              B.data_0th[:,np.newaxis,:,np.newaxis],
                                              A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                             ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                       np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                       2*A.dims_cvs),
                       data_1st_r = np.einsum('jpkqz,plqm->kljmz',
                                              B.data_1st[:,np.newaxis,:,np.newaxis,:],
                                              A.data_0th[np.newaxis,:,np.newaxis,:]
                                             ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                       np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                       2*B.dims_cvs),
                       data_0th = np.einsum('jpkq,plqm->kljm',
                                            B.data_0th[:,np.newaxis,:,np.newaxis],
                                            A.data_0th[np.newaxis,:,np.newaxis,:]
                                           ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                     np.prod((A.dims_fls[1], B.dims_fls[0]))),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], B.dims_fls[1]],
                                   [A.dims_fls[1], B.dims_fls[0]]])

    else:
        return QGsuper(data_2nd_l = A.data_2nd*B.data_0th,
                       data_2nd_r = A.data_0th*B.data_2nd,
                       data_2nd_m = np.einsum('j,k->jk',A.data_1st,B.data_1st),
                       data_1st_l = A.data_1st*B.data_0th,
                       data_1st_r = A.data_0th*B.data_1st,
                       data_0th = A.data_0th*B.data_0th,
                       dims_cvs = A.dims_cvs)