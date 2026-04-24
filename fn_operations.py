import numpy.typing as npt
import qgauss
import numpy as np
from scipy import linalg as la

from .qgstate import QGstate
from .qgoper import QGoper
from .qgsuper import QGsuper
from .fn_utilities import exp_integrator_phi_function

__all__ = ['expect','commutator','ASp_transform']


def expect(oper: QGoper,
           state: QGstate, 
           ) -> complex:
    """
    ---- Prodcedure ----
    Expectation value of an operator "Q" for a specified state "ρ". For a system which is FLS only this is the usual 
    trace over a product of matrices:
        expect = tr_FLS[ρ.Q] = tr_FLS[ρ @ Q]
    For a system which is CVS only, the trace is replaced by a product of moments of ρ and coefficients of Q:
        expect = tr_CVS[ρ.Q] = Exp[ν]*( ½*tr[Σ.O(2)] + ½*μ.O(2).μ + μ.O(1) + O(0) )
    For a mixed system, the expectation value is calculated by taking the matrix product between ρ and Q at the FLS
    level, and then summing over the different CVS expectation values:
        expect = sum_{j,k} tr_CVS[ρ[j,k].Q[k,j]]

    ---- Parameters ----
    oper : QGoper
    state : QGstate

    ---- Returns ----
    expect : number
    """
    # Check that state and oper have the same internal structure/dimensions
    if (state.dims_fls != oper.dims_fls) or (state.dims_cvs != oper.dims_cvs):
        raise ValueError("State and operator have different dimensions.")

    # Determine how to calculate expectation value depending on if both state and operator are FLS and/or CVS
    if state.isfls and not state.iscvs:
        return np.trace(state.data_0th @ oper.data_0th)
    
    # When CVS components are present, we must check that the state is integrable
    elif state.iscvs and state.isintegrable:
        if not state.isfls:
            return _expect_cv(state, oper)
        elif state.isfls:
            return sum(_expect_cv(state[j,k], oper[k,j])
                       for j,k in zip(np.prod(state.dims_fls[0]),np.prod(state.dims_fls[1])))
    
    # CVS-components are determined to not be integrable, and hence trace cannot be performed
    else:
        raise ValueError("State is not integrable, and hence the expectation value cannot be computed.")


def _expect_cv(state: QGstate, oper: QGoper):
    """ Private helper function for expect to calculate expectation value for CV only state and operator. """
    return state.data_0th*(0.5*np.trace(state.data_2nd @ oper.data_2nd)
                           + 0.5*(state.data_1st @ oper.data_2nd @ state.data_1st)
                           + state.data_1st @ oper.data_1st
                           + oper.data_0th
                           )


def commutator(A: QGoper, 
               B: QGoper
               ) -> QGoper:
    """
    ---- Prodcedure ----
    Commutator of a pair of operators, [A,B]. Both operators must be FLS or CVS only; the handling of mixed operators is 
    currently not implemented. If both operators are FLS only then the commuator is simply the usual
        [A,B]_FLS = A.B - B.A
    If the pair are CVS only operators, then the commutators are evaluated using the known commutation relation for
    the quadrature operators.
        [r_j,r_k] = i*Ω_jk
    Representing the operators as
        K = ½r.OK(2).r + r.OK(1) + OK(0) for K = A,B
    the CVS commutators are then evaluated using the following expression
        [A,B]_CVS = ½r.(i*OA(2).Ω.OB(2) - i*OB(2).Ω.OA(2)).r 
                    + r.(i*OA(2).Ω.OB(1) - i*OB(2).Ω.OA(1)) + i*OA(1).Ω.OB(1)

    ---- Parameters ----
    A : QGoper
    B : QGoper

    ---- Returns ----
    commutator : QGoper
    """
    # Check that state and oper have the same internal structure/dimensions
    if (A.dims_fls != B.dims_fls) or (A.dims_cvs != B.dims_cvs):
        raise ValueError("Operators have different dimensions.")
    
    # Apply for commutation relations for pairs of FLS-only or CVS-only operators
    if (A.isfls and not A.iscvs) and (B.isfls and not B.iscvs):
        return A*B - B*A
    
    elif (not A.isfls and A.iscvs) and (not B.isfls and B.iscvs):
        symform = A.symform
        return QGoper(data_2nd = 1j*(A.data_2nd @ symform @ B.data_2nd
                                     - B.data_2nd @ symform @ A.data_2nd),
                      data_1st = 1j*(A.data_2nd @ symform @ B.data_1st
                                     + A.data_1st @ symform @ B.data_2nd),
                      data_0th = 1j*(A.data_1st @ symform @ B.data_1st),
                      dims_cvs = A.dims_cvs
                      )
    
    # Commutators of mixed FLS/CVS operators not yet implemented
    else:
        NotImplemented


def ASp_transform(input: QGstate | QGoper | QGsuper,
                  gen_oper: QGoper = None
                  ) -> QGstate | QGoper | QGsuper:
    """
    ---- Prodcedure ----
    Note: This is a test function, and may be incorporated as a class method in future.
    Apply an arbitrary symplectic-affine/canonical transformation to a state, operator, or superoperator, equivalent to 
    a unitary operator. The input can be a mixed CVS/FLS object, but the generators must be CVS-only operators. For a 
    symplectic-affine transformation with generator X = ½r.X(2).r + r.X(1), the quadrature operators will transform as:
        r -> U*.r.U = S.r + d  where  U = exp[-i*X], and 
        S = exp[Ω.X(2)], d(t) = (1/Ω.X(2)).(exp[t*Ω.X(2)] - id).(Ω.X(1)).
    For an operator Q = ½r.Q(2).r + r.Q(1) + Q(0), this transformation will take the form.
        Q -> ½r.S^T.Q(2).S.r + r.S^T.(Q(2).d + Q(1)) + (½d.Q(2).d + d.Q(1) + O(0))
    The above expressions can be used to apply these transformations to operators and superoperators, taking advantage
    of the fact that it is ultimately just a unitary transformation. For states, the evolution is:
        ρ -> U.ρ.U* = exp[-i[X,·]].ρ  where  U = exp[-i*X].
    This is equivalent to the following evolution of the moments of ρ(0)
        Σ -> S.Σ.S^T,   
        μ -> S.μ + d
    where the transformations S and d are as before. For objects with an FLS part, the transformation is applied to
    each sub-component of the total object.

    ---- Parameters ----
    input : QGstate or QGoper or QGsuper
        Object on which to apply the symplectic transformations.
    gen_oper : QGoper
        QGoper that will act as the generator of the unitary transformations, which is applied as a symplectic-affine
        transformation on the data components of input.

    ---- Output ----
    output : QGstate or QGoper or QGsuper
        Object obtained after application of the symplectic transformation(s).
    """
    if gen_oper is None:
        return input
    if gen_oper.isfls:
        raise ValueError("Generator of symplectic-affine transformation cannot have FLS component.")
    if gen_oper.dims_cvs != input.dims_cvs:
        raise ValueError("CV component of input and generator of the symplectic-affine transformation" \
                         " have different dimensions.")
    
    # Construct elements of the symplectic-affine transformation
    S = la.expm(gen_oper.symform @ gen_oper.data_2nd)
    d = exp_integrator_phi_function(gen_oper.symform @ gen_oper.data_2nd) @ (gen_oper.symform @ gen_oper.data_1st)

    if isinstance(input, QGstate):
        if not input.isfls:
            _out_data_2nd = S @ input.data_2nd @ np.transpose(S)
            _out_data_1st = S @ input.data_1st + d
        else:
            _out_data_2nd = np.array([[(S @ input[j,k].data_2nd @ np.transpose(S))
                                       for k in range(input.shape_0th[1])]
                                       for j in range(input.shape_0th[0])])
            _out_data_1st = np.array([[(S @ input[j,k].data_1st + d)
                                       for k in range(input.shape_0th[1])]
                                       for j in range(input.shape_0th[0])])

        return QGstate(data_2nd = _out_data_2nd,
                       data_1st = _out_data_1st,
                       data_0th = input.data_0th,
                       dims_fls = input.dims_fls,
                       dims_cvs = input.dims_cvs
                       )

    elif isinstance(input, QGoper):
        if not input.isfls:
            _out_data_2nd = np.transpose(S) @ input.data_2nd @ S
            _out_data_1st = (0.5*(np.transpose(S) @ input.data_2nd @ d)
                             + 0.5*(d @ input.data_2nd @ S) 
                             + (input.data_1st @ S))
            _out_data_0th = (0.5*(d @ input.data_2nd @ d) 
                             + (input.data_1st @ d) 
                             + input.data_0th)
        else:
            _out_data_2nd = np.array([[(np.transpose(S) @ input.data_2nd[j,k] @ S)
                                       for k in range(input.shape_0th[1])]
                                       for j in range(input.shape_0th[0])])
            _out_data_1st = np.array([[(0.5*(np.transpose(S) @ input.data_2nd[j,k] @ d)
                                        + 0.5*(d @ input.data_2nd[j,k] @ S)
                                        + (input.data_1st[j,k] @ S))
                                       for k in range(input.shape_0th[1])]
                                       for j in range(input.shape_0th[0])])
            _out_data_0th = np.array([[(0.5*(d @ input.data_2nd[j,k] @ d)
                                        + (input.data_1st[j,k] @ d)
                                        + input.data_0th[j,k])
                                       for k in range(input.shape_0th[1])]
                                       for j in range(input.shape_0th[0])])
    
        return QGoper(data_2nd = _out_data_2nd,
                      data_1st = _out_data_1st,
                      data_0th = _out_data_0th,
                      dims_fls = input.dims_fls,
                      dims_cvs = input.dims_cvs
                      )
    
    elif isinstance(input, QGsuper):
        if not input.isfls:
            _out_data_2nd_l = np.transpose(S) @ input.data_2nd_l @ S
            _out_data_2nd_r = np.transpose(S) @ input.data_2nd_r @ S
            _out_data_2nd_m = np.transpose(S) @ input.data_2nd_m @ S
            _out_data_1st_l = (0.5*(np.transpose(S) @ input.data_2nd_l @ d)
                               + 0.5*(d @ input.data_2nd_l @ S) 
                               + (d @ input.data_2nd_m @ S) 
                               + (input.data_1st_l @ S))
            _out_data_1st_r = (0.5*(np.transpose(S) @ input.data_2nd_r @ d)
                               + 0.5*(d @ input.data_2nd_r @ S) 
                               + (np.transpose(S) @ input.data_2nd_m @ d) 
                               + (input.data_1st_r @ S))
            _out_data_0th = (0.5*(d @ input.data_2nd_l @ d) 
                             + 0.5*(d @ input.data_2nd_r @ d)
                             + (d @ input.data_2nd_m @ d) 
                             + (input.data_1st_l @ d) 
                             + (input.data_1st_r @ d)
                             + input.data_0th)
        else:
            _out_data_2nd_l = np.array([[(np.transpose(S) @ input.data_2nd_l[j,k] @ S)
                                         for k in range(input.shape_0th[1])]
                                         for j in range(input.shape_0th[0])])
            _out_data_2nd_r = np.array([[(np.transpose(S) @ input.data_2nd_r[j,k] @ S)
                                         for k in range(input.shape_0th[1])]
                                         for j in range(input.shape_0th[0])])
            _out_data_2nd_m = np.array([[(np.transpose(S) @ input.data_2nd_m[j,k] @ S)
                                         for k in range(input.shape_0th[1])]
                                         for j in range(input.shape_0th[0])])
            _out_data_1st_l = np.array([[(0.5*(np.transpose(S) @ input.data_2nd_l[j,k] @ d)
                                          + 0.5*(d @ input.data_2nd_l[j,k] @ S)
                                          + (d @ input.data_2nd_m[j,k] @ S)
                                          + (input.data_1st_l[j,k] @ S))
                                         for k in range(input.shape_0th[1])]
                                         for j in range(input.shape_0th[0])])
            _out_data_1st_r = np.array([[(0.5*(np.transpose(S) @ input.data_2nd_r[j,k] @ d)
                                          + 0.5*(d @ input.data_2nd_r[j,k] @ S)
                                          + (np.transpose(S) @ input.data_2nd_m[j,k] @ d)
                                          + (input.data_1st_r[j,k] @ S))
                                         for k in range(input.shape_0th[1])]
                                         for j in range(input.shape_0th[0])])
            _out_data_0th = np.array([[(0.5*(d @ input.data_2nd_l[j,k] @ d)
                                        + 0.5*(d @ input.data_2nd_r[j,k] @ d)
                                        + (d @ input.data_2nd_m[j,k] @ d)
                                        + (input.data_1st_l[j,k] @ d)
                                        + (input.data_1st_r[j,k] @ d)
                                        + input.data_0th[j,k])
                                         for k in range(input.shape_0th[1])]
                                         for j in range(input.shape_0th[0])])
        
        return QGsuper(data_2nd_l = _out_data_2nd_l,
                       data_2nd_r = _out_data_2nd_r,
                       data_2nd_m = _out_data_2nd_m,
                       data_1st_l = _out_data_1st_l,
                       data_1st_r = _out_data_1st_r,
                       data_0th = _out_data_0th,
                       dims_fls = input.dims_fls,
                       dims_cvs = input.dims_cvs
                       )