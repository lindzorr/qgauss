import math
import numpy as np

from .qgoper import QGoper
from .qgsuper import QGsuper
from ..calc.utilities import fls_size

__all__ = ['spre','spost','sprepost',
           'commutator_super','anticommutator_super',
           'dissipator','coherent','lindbladian'
          ]

""" Constructors used to create superoperators from operators. """

def dissipator(A: QGoper, 
               B: QGoper = None
              ) -> QGsuper:
    # Lindblad dissipation term: D[A,B](ρ) = A.ρ.B† - ½[B†.A,ρ]_+, where D[A,A] = D[A]
    if B is None:
        B = A
    Bd = B.dag()
    BdA = Bd*A
    return sprepost(A,Bd) - (1/2)*spre(BdA) - (1/2)*spost(BdA)


def coherent(H: QGoper) -> QGsuper:
    # Lindblad coherent evolution/von Neumann term: -i[H,ρ] = -i(H.ρ - ρ.H)
    return -1j*(spre(H) - spost(H))


def lindbladian(H: QGoper = None, 
                c_ops: list[QGoper] = None
               ) -> QGsuper:
    # Lindblad superoperator, L(ρ) = -i[H,ρ] + Σ_{c_ops} D[c_ops](ρ)
    if c_ops is None:
        c_ops = []
    elif isinstance(c_ops, QGoper):
        c_ops = [c_ops]
    if H is None and len(c_ops) == 0:
        raise ValueError("Must provide either H or a non-empty c_ops.")
    if H is not None:
        L = -1j*(spre(H) - spost(H))
    else:
        L = 0
        
    L += sum(dissipator(c) for c in c_ops)
    return L


def commutator_super(H: QGoper) -> QGsuper:
    # Commutator superoperator: [H,ρ] = H.ρ - ρ.H
    return (spre(H) - spost(H))


def anticommutator_super(H: QGoper) -> QGsuper:
    # Anti-commutator superoperator: [H,ρ]_+ = H.ρ + ρ.H
    return (spre(H) + spost(H))


def spost(A: QGoper) -> QGsuper:
    # Superoperator representing post/right-multiplication of state by an operator: ρ.A
    if A.isfls:
        if A.dims_fls[0] != A.dims_fls[1]:
            raise ValueError("spost is only defined for square FLS operators " \
                             "(dims_fls[0] must equal dims_fls[1]).")
        return QGsuper(data_2nd_r = \
                       np.einsum('jpkqyz,plqm->kljmyz',
                                 A.data_2nd[:,np.newaxis,:,np.newaxis,:,:],
                                 np.identity(math.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           2*A.dims_cvs,
                                           2*A.dims_cvs),
                       data_1st_r = \
                       np.einsum('jpkqz,plqm->kljmz',
                                 A.data_1st[:,np.newaxis,:,np.newaxis,:],
                                 np.identity(math.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           2*A.dims_cvs),
                       data_0th = \
                       np.einsum('jpkq,plqm->kljm',
                                 A.data_0th[:,np.newaxis,:,np.newaxis],
                                 np.identity(math.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           fls_size(A.dims_fls[0], A.dims_fls[1])),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], A.dims_fls[1]],
                                   [A.dims_fls[0], A.dims_fls[1]]])
    else:
        return QGsuper(data_2nd_r = A.data_2nd,
                       data_1st_r = A.data_1st,
                       data_0th = A.data_0th.copy(),
                       dims_cvs = A.dims_cvs)


def spre(A: QGoper) -> QGsuper:
    # Superoperator representing pre/left-multiplication of state by an operator: A.ρ
    if A.isfls:
        if A.dims_fls[0] != A.dims_fls[1]:
            raise ValueError("spre is only defined for square FLS operators " \
                             "(dims_fls[0] must equal dims_fls[1]).")
        return QGsuper(data_2nd_l = \
                       np.einsum('jpkq,plqmyz->jlkmyz',
                                 np.identity(math.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                 A.data_2nd[np.newaxis,:,np.newaxis,:,:,:]
                                 ).reshape(fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           2*A.dims_cvs,
                                           2*A.dims_cvs),
                       data_1st_l = \
                       np.einsum('jpkq,plqmz->jlkmz',
                                 np.identity(math.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                 A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                 ).reshape(fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           2*A.dims_cvs),
                       data_0th = \
                       np.einsum('jpkq,plqm->jlkm',
                                 np.identity(math.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                 A.data_0th[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], A.dims_fls[1]),
                                           fls_size(A.dims_fls[0], A.dims_fls[1])),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], A.dims_fls[1]],
                                   [A.dims_fls[0], A.dims_fls[1]]])                
    else:
        return QGsuper(data_2nd_l = A.data_2nd,
                       data_1st_l = A.data_1st,
                       data_0th = A.data_0th.copy(),
                       dims_cvs = A.dims_cvs)


def sprepost(A: QGoper, 
             B: QGoper
            ) -> QGsuper:
    # Superoperator representing pre/left and post-right-multiplication of a
    # state by an operator: A.ρ.B
    # Requires that A and B have the same dimensions.
    if (A.isfls != B.isfls) or (A.iscvs != B.iscvs): 
        raise ValueError("sprepost requires that A and B act on the same Hilbert space.")
    
    if A.dims_cvs != B.dims_cvs:
        raise ValueError("sprepost requires that A and B act on the same " \
                         "CVS space (A.dims_cvs must equal B.dims_cvs).")
    if ((A.is2nd and B.is2nd) or
        (A.is2nd and B.is1st) or
        (A.is1st and B.is2nd)
        ):
        raise ValueError("Inputs result in superoperator which is beyond " \
                    "quadratic/bilinear order in the quadrature operators.")

    if A.isfls and B.isfls:
        if (A.dims_fls[0] != A.dims_fls[1]) or (B.dims_fls[0] != B.dims_fls[1]):
            raise ValueError("sprepost is only defined for square FLS operators " \
                             "(dims_fls[0] must equal dims_fls[1]).")
        if A.dims_fls != B.dims_fls:
            raise ValueError("sprepost requires that A and B act on the same " \
                             "FLS space (A.dims_fls must equal B.dims_fls).")
        return QGsuper(data_2nd_l = \
                       np.einsum('jpkq,plqmyz->kljmyz',
                                 B.data_0th[:,np.newaxis,:,np.newaxis],
                                 A.data_2nd[np.newaxis,:,np.newaxis,:,:,:]
                                 ).reshape(fls_size(A.dims_fls[0], B.dims_fls[1]),
                                           fls_size(A.dims_fls[1], B.dims_fls[0]),
                                           2*A.dims_cvs,
                                           2*A.dims_cvs),
                       data_2nd_r = \
                       np.einsum('jpkqyz,plqm->kljmyz',
                                 B.data_2nd[:,np.newaxis,:,np.newaxis,:,:],
                                 A.data_0th[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], B.dims_fls[1]),
                                           fls_size(A.dims_fls[1], B.dims_fls[0]),
                                           2*B.dims_cvs,
                                           2*B.dims_cvs),
                       data_2nd_m = \
                       np.einsum('jpkqz,plqmy->kljmyz',
                                 B.data_1st[:,np.newaxis,:,np.newaxis,:],
                                 A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                 ).reshape(fls_size(A.dims_fls[0], B.dims_fls[1]),
                                           fls_size(A.dims_fls[1], B.dims_fls[0]),
                                           2*A.dims_cvs,
                                           2*B.dims_cvs),
                       data_1st_l = \
                       np.einsum('jpkq,plqmz->kljmz',
                                 B.data_0th[:,np.newaxis,:,np.newaxis],
                                 A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                 ).reshape(fls_size(A.dims_fls[0], B.dims_fls[1]),
                                           fls_size(A.dims_fls[1], B.dims_fls[0]),
                                           2*A.dims_cvs),
                       data_1st_r = \
                       np.einsum('jpkqz,plqm->kljmz',
                                 B.data_1st[:,np.newaxis,:,np.newaxis,:],
                                 A.data_0th[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], B.dims_fls[1]),
                                           fls_size(A.dims_fls[1], B.dims_fls[0]),
                                           2*B.dims_cvs),
                       data_0th = \
                       np.einsum('jpkq,plqm->kljm',
                                 B.data_0th[:,np.newaxis,:,np.newaxis],
                                 A.data_0th[np.newaxis,:,np.newaxis,:]
                                 ).reshape(fls_size(A.dims_fls[0], B.dims_fls[1]),
                                           fls_size(A.dims_fls[1], B.dims_fls[0])),
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