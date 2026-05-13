import numbers
import numpy.typing as npt
import qgauss
import numpy as np

from .qgstate import QGstate
from .qgoper import QGoper

__all__ = ['vacuum','thermal','displaced','sm_squeeze','tm_squeeze',
           'qubit_up','qubit_down','qubit_plus','qubit_minus',
           'qubit_plus','qubit_minus','basis_state',
           'destroy','create','position','momentum',
           'one','identity','identity_cvs','num',
           'qeye','identity_fls','qzero','basis_oper',
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
    return QGstate(data_2nd = (1/2)*np.identity(2*N), 
                   dims_cvs = N)

def thermal(*nth: float | list[float] | npt.NDArray[float]) -> QGstate:
    # N-mode thermal state, with occupancies "nth"
    if not nth:
        _nth = 0
    elif len(nth) == 1 and isinstance(nth[0], (numbers.Number, np.number)):
        _nth = np.asarray([nth[0]])
    elif len(nth) == 1 and isinstance(nth[0], list | np.ndarray):
        _nth = np.asarray(nth[0])
    return QGstate(data_2nd = np.kron(np.diag(_nth) + (1/2)*np.identity(len(_nth)),
                                      np.identity(2)),
                   dims_cvs = len(_nth))

def displaced(alpha: complex | list[float] | npt.NDArray[float] = None
             ) -> QGstate:
    # Single mode coherent/displaced vacuum state
    if alpha is None:
        _q, _p = 0, 0
    elif isinstance(alpha, (numbers.Number, np.number)):
        _q, _p = np.real(alpha), np.imag(alpha)
    elif isinstance(alpha, np.ndarray | list) and len(alpha) == 2:
        _q, _p = alpha[0], alpha[1]
    return QGstate(data_2nd = (1/2)*np.identity(2),
                   data_1st = np.array([_q,_p]),
                   dims_cvs = 1)

def sm_squeeze(sqz: complex | list[float] | npt.NDArray[float] = None
              ) -> QGstate:
    # Single-mode squeezed state
    if sqz is None:
        _r, _t = 0, 0
    elif isinstance(sqz, (numbers.Number, np.number)):
        _r, _t = np.abs(sqz), np.angle(sqz)
    elif isinstance(sqz, np.ndarray | list) and len(sqz) == 2:
        _r, _t = sqz[0], sqz[1]

    _cov = \
    (1/2)*np.array([[np.cosh(2*_r) + np.cos(_t)*np.sinh(2*_r), -np.sin(_t)*np.sinh(2*_r)],
                    [-np.sin(_t)*np.sinh(2*_r), np.cosh(2*_r) - np.cos(_t)*np.sinh(2*_r)]])
    
    return QGstate(data_2nd = _cov,
                   dims_cvs = 1)

def tm_squeeze(sqz: complex | list[float] | npt.NDArray[float] = None
              ) -> QGstate:
    # Two-mode squeezed state
    if sqz is None:
        _r, _t = 0, 0
    elif isinstance(sqz, (numbers.Number, np.number)):
        _r, _t = np.abs(sqz), np.angle(sqz)
    elif isinstance(sqz, np.ndarray | list) and len(sqz) == 2:
        _r, _t = sqz[0], sqz[1]

    _cov = \
    (1/2)*np.array([[np.cosh(2*_r), 0, -np.cos(_t)*np.sinh(2*_r), -np.sin(_t)*np.sinh(2*_r)],
                    [0, np.cosh(2*_r), -np.sin(_t)*np.sinh(2*_r), +np.cos(_t)*np.sinh(2*_r)],
                    [-np.cos(_t)*np.sinh(2*_r), -np.sin(_t)*np.sinh(2*_r), np.cosh(2*_r), 0],
                    [-np.sin(_t)*np.sinh(2*_r), +np.cos(_t)*np.sinh(2*_r), 0, np.cosh(2*_r)]])
    
    return QGstate(data_2nd = _cov,
                   dims_cvs = 2)

def qubit_up() -> QGstate:
    # Density matrix for qubit up/excited-state
    return QGstate(data_0th = np.array([[1,0],[0,0]]),
                   dims_fls = [[2],[2]])

def qubit_down() -> QGstate:
    # Density matrix for qubit down/ground-state
    return QGstate(data_0th = np.array([[0,0],[0,1]]),
                   dims_fls = [[2],[2]])

def qubit_plus() -> QGstate:
    # Density matrix for qubit plus-state, (|0>+|1>)/sqrt(2)
    return QGstate(data_0th = (1/2)*np.array([[1,1],[1,1]]),
                   dims_fls = [[2],[2]])

def qubit_minus() -> QGstate:
    # Density matrix for qubit minus-state, (|0>-|1>)/sqrt(2)
    return QGstate(data_0th = (1/2)*np.array([[1,-1],[-1,1]]),
                   dims_fls = [[2],[2]])

def qubit_right() -> QGstate:
    # Density matrix for qubit right-state, (|0>+i|1>)/sqrt(2)
    return QGstate(data_0th = (1/2)*np.array([[1,1j],[-1j,1]]),
                   dims_fls = [[2],[2]])

def qubit_lefft() -> QGstate:
    # Density matrix for qubit left-state, (|0>-i|1>)/sqrt(2)
    return QGstate(data_0th = (1/2)*np.array([[1,-1j],[1j,1]]),
                   dims_fls = [[2],[2]])

def basis_state(m: int, N: int = 1) -> QGoper:
    # N-by-N state corresponding to the outer product of a single basis vector, 
    # |m><m|. To obey the convention used here for FLs operators, |0><0| has a 
    # one in the lower-right corner, while |N-1><N-1| is in the upper-left 
    # corner of the data matrix.
    _data = np.zeros((N, N))
    _data[N-m-1,N-m-1] = 1
    return QGstate(data_0th = _data,
                   dims_fls = [[N],[N]])

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
    # Alias of identity
    return QGoper(data_0th = np.array([1.]), 
                  dims_cvs = N)

def destroy(M: int = 1, N: int = None) -> QGoper:
    # Single mode annihilation operator for the Mth mode in an N-mode 
    # continuous-variable system.
    if N is None: 
        N = M
    _data = np.zeros(2*N, dtype = complex)
    _data[2*M-2:2*M] = np.array([1,1j])/np.sqrt(2)
    return QGoper(data_1st = _data, 
                  dims_cvs = N)

def create(M: int = 1, N: int = None) -> QGoper:
    # Single mode creation operator for the Mth mode in an N-mode 
    # continuous-variable system.
    if N is None: 
        N = M
    _data = np.zeros(2*N, dtype = complex)
    _data[2*M-2:2*M] = np.array([1,-1j])/np.sqrt(2)
    return QGoper(data_1st = _data, 
                  dims_cvs = N)

def position(M: int = 1, N: int = None) -> QGoper:
    # Single mode position operator for the Mth mode in an N-mode 
    # continuous-variable system.
    if N is None: 
        N = M
    _data = np.zeros(2*N, dtype = complex)
    _data[2*M-2:2*M] = np.array([1,0])
    return QGoper(data_1st = _data, 
                  dims_cvs = N)
    
def momentum(M: int = 1, N: int = None) -> QGoper:
    # Single mode momentum operator for the Mth mode in an N-mode 
    # continuous-variable system.
    if N is None: 
        N = M
    _data = np.zeros(2*N, dtype = complex)
    _data[2*M-2:2*M] = np.array([0,1])
    return QGoper(data_1st = _data, 
                  dims_cvs = N)

def num(M: int = 1, N: int = None) -> QGoper:
    # Single mode number operator for the Mth mode in an N-mode 
    # continuous-variable system.
    if N is None: 
        N = M
    _data = np.zeros((2*N,2*N), dtype = complex)
    _data[2*M-2:2*M,2*M-2:2*M] = np.array([[1,0],[0,1]])
    return QGoper(data_2nd = _data, 
                  data_0th = -1/2, 
                  dims_cvs = N)

def identity_fls(dims: int | list[int] | npt.NDArray[int] = 1) -> QGoper:
    # N-by-N identity operator for a finite-level system, where N may an 
    # integer or list of integers
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

def basis_oper(m: int, N: int = None) -> QGoper:
    # N-by-N operator corresponding to the outer product of a single basis 
    # vector, |m><m|. To obey the convention used here for FLs operators, |0><0|
    # has a one in the lower-right corner, while |N-1><N-1| is in the upper-left 
    # corner of the data matrix.
    if N is None: 
        N = m
    _data = np.zeros((N, N))
    _data[N-m-1,N-m-1] = 1
    return QGoper(data_0th = _data, 
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
    # Spin operator with total spin "j", where the string "comp" denotes the 
    # component of the spin, and only takes values ['+','-','x','y','z','tot'].
    if j < 0:
        raise TypeError("The total spin j must be a non-negative integer.")
    if 2*j != np.fix(2*j):
        raise TypeError("The total spin j can only take integer or half-integer values.")

    if comp == '+':
        _data = np.diag([np.sqrt(j*(j+1)-m*(m+1))
                         for m in np.arange(j-1,-j-1,-1)], k=1)
    elif comp == '-':
        _data = np.diag([np.sqrt(j*(j+1)-m*(m-1))
                         for m in np.arange(j,-j,-1)], k=-1)
    elif comp == 'x':
        _data = (1/2)*(np.diag([np.sqrt(j*(j+1)-m*(m+1))
                                for m in np.arange(j-1,-j-1,-1)], k=1)
                       + np.diag([np.sqrt(j*(j+1)-m*(m-1))
                                  for m in np.arange(j,-j,-1)], k=-1))
    elif comp == 'y':
        _data = -(1j/2)*(np.diag([np.sqrt(j*(j+1)-m*(m+1)) 
                                  for m in np.arange(j-1,-j-1,-1)], k=1)
                         - np.diag([np.sqrt(j*(j+1)-m*(m-1)) 
                                    for m in np.arange(j,-j,-1)], k=-1))
    elif comp == 'z':
        _data = np.diag([m for m in np.arange(j,-j-1,-1)])
    elif comp == 'tot':
        _data = j*(j+1)*np.identity(int(2*j+1))
    else:
        raise TypeError("A valid component of the spin must be provided. " \
        "Choose one of ['+','-','x','y','z','tot'].")
                        
    return QGoper(data_0th = _data,
                  dims_fls = [[int(2*j+1)],[int(2*j+1)]])