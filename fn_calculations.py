import sys
import types
import warnings
import functools

import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np
from scipy import linalg as la

from .qgstate import *
from .qgoper import *
from .qgsuper import *

__all__ = ['expect','commutator','ASp_transform','unitary_evolve','lindblad_evolve']


def expect(state: QGstate, 
           oper: QGoper
           ):
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
    state : QGstate
    oper : QGoper

    ---- Returns ----
    expect : number
    """
    # Check that state and oper have the same internal structure/dimensions
    if (state.dims_fls != oper.dims_fls) or (state.dims_cvs != oper.dims_cvs):
        raise ValueError("State and operator have different dimensions.")

    # Determine how to calculate expectation value depending on if both state and operator are FLS and/or CVS
    if (state.isfls and not state.iscvs):
        return np.trace(state.data_0th @ oper.data_0th)
    
    elif (not state.isfls and state.iscvs):
        return _cvs_expect(state, oper)

    elif (state.isfls and state.iscvs):
        return sum(_cvs_expect(state[j,k], oper[k,j]) 
                   for j,k in zip(np.prod(state.dims_fls[0]),np.prod(state.dims_fls[1])))


def _cvs_expect(state: QGstate, oper: QGoper):
    """ Private helper function for expect to calculate expectation value for CV only state and operator. """
    return state.data_0th*(0.5*np.trace(state.data_2nd @ oper.data_2nd)
                           + 0.5*(state.data_1st @ oper.data_2nd @ state.data_1st)
                           + state.data_1st @ oper.data_1st
                           + oper.data_0th
                           )


def commutator(operA: QGoper, 
               operB: QGoper
               ) -> QGoper:
    """
    ---- Prodcedure ----
    Commutator of a pair of operators, operA=QA  and operaB=QB, [QA,QB]. Both operators must be FLS or CVS only; 
    the handling of mixed operators is currently not implemented. If both operators are FLS only then the commuator is
    simply the usual
        [QA,QB]_FLS = QA.QB - QB.QA
    If the pair are CVS only operators, then the commutators are evaluated using the known commutation relation for
    the quadrature operators.
        [r_j,r_k] = i*Ω_jk
    Representing the operators as
        QK = ½r.OK(2).r + r.OK(1) + OK(0) for K = A,B
    the CVS commutators are then evaluated using the following expression
        [QA,QB]_CVS = ½r.(i*OA(2).Ω.OB(2) - i*OB(2).Ω.OA(2)).r 
                        + r.(i*OA(2).Ω.OB(1) - i*OB(2).Ω.OA(1)) + i*OA(1).Ω.OB(1)

    ---- Parameters ----
    operA : QGoper
    operB : QGoper

    ---- Returns ----
    commutator : QGoper
    """
    # Check that state and oper have the same internal structure/dimensions
    if (operA.dims_fls != operB.dims_fls) or (operA.dims_cvs != operB.dims_cvs):
        raise ValueError("Operators have different dimensions.")
    
    # Apply for commutation relations for pairs of FLS-only or CVS-only operators
    if (operA.isfls and not operA.iscvs) and (operB.isfls and not operB.iscvs):
        return operA * operB - operB * operA
    
    elif (not operA.isfls and operA.iscvs) and (not operB.isfls and operB.iscvs):
        symform = operA.symform
        return QGoper(data_2nd = 1j*(operA.data_2nd @ symform @ operB.data_2nd
                                     - operB.data_2nd @ symform @ operA.data_2nd),
                      data_1st = 1j*(operA.data_2nd @ symform @ operB.data_1st
                                     + operA.data_1st @ symform @ operB.data_2nd),
                      data_0th = 1j*(operA.data_1st @ symform @ operB.data_1st),
                      dims_cvs = operA.dims_cvs
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
    a unitary operator. The input can be a mixed CVS/FLS object, but the generators must be CVS only operators. For a 
    symplectic-affine transformation with generator X = ½r.X(2).r + r.X(1), the quadrature operators will transform as:
        r -> U*.r.U = S.r + d  where  U = exp[-i*X], and 
        S = exp[Ω.X(2)], d(t) = (1/Ω.X(2)).(exp[t*Ω.X(2)] - id).(Ω.X(1)).
    For an operator Q = ½r.Q(2).r + r.Q(1) + Q(0), this transformation will take the form.
        Q -> ½r.S^T.Q(2).S.r + r.S^T.(Q(2).d + Q(1)) + (½d.Q(2).d + d.Q(1) + O(0))
    The above expressions can be used to apply these transformations to operators and superoperators, taking advantage
    of the fact that it is ultimately just a unitary transformation. For the details about the evolution of states, 
    see the procedure in the "unitary_evolve" function. For objects with an FLS part, the transformation is applied to
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
    d = _exp_integrator_phi_function(gen_oper.symform @ gen_oper.data_2nd) @ (gen_oper.symform @ gen_oper.data_1st)

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
    

def _exp_integrator_phi_function(X, k=1):
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


def unitary_evolve(rho0: QGstate, 
                   H: QGoper, 
                   t = 1
                   ) -> QGstate:
    """
    ---- Prodcedure ----
    Apply a unitary operation to QGstate, equivalent to a symplectic-affine transformation. The time evolution is
        ρ(t) = U(t).ρ(0).U*(t) = exp[-i*t*[H,·]].ρ(0)  where  U(t) = exp[-i*H*t].
    This is equivalent to the following evolution of the moments of ρ(0)
        Σ(t) = S(t).Σ(0).S(t)^T,   
        μ(t) = S(t).μ(0) + d(t)
    where for the Hamiltonian H = ½r.H(2).r + r.H(1) + H(0), and A = Ω.H(2),
        S(t) = exp[t*A],    
        d(t) = (1/A).(exp[t*A] - id).(Ω.H(1)).
    For ρ0 with an FLS component, the above expressions are applied individually to each CVS sub-component. These are 
    the exact solutions to the differential equations for the moments:
        dΣ(t)/dt = A.Σ(t) + Σ(t).A^T,
        dμ(t)/dt = A.μ(t) + d(t).
    
    ---- Parameters ----
    rho0 : QGstate
        Object on which to apply the symplectic-affine transformations.
    H : QGoper
        Time-independent Hamiltonian that will control evolution of the state. Must be CVS only and Hermitian.
    t : number
        Time at which to stop calculate the future state. Time starts at t=0, default for the future time is t=1.

    ---- Output ----
    output : QGstate
        The state evolved under the Hamiltonian at the give time.
    """
    if H.isfls:
        raise ValueError("Generator of unitary evolution cannot have FLS component.")
    if H.dims_cvs != rho0.dims_cvs:
        raise ValueError("CV component of state and Hamiltonian have different dimensions.")
    
    # Construct elements of the unitary transformation
    A = H.symform @ H.data_2nd
    S = la.expm(t*A)
    d = t * _exp_integrator_phi_function(t*A) @ (H.symform @ H.data_1st)

    if not rho0.isfls:
        _out_data_2nd = S @ rho0.data_2nd @ np.transpose(S)
        _out_data_1st = S @ rho0.data_1st + d
    else:
        _out_data_2nd = np.array([[(S @ rho0[j,k].data_2nd @ np.transpose(S)) 
                                   for k in range(rho0.shape_0th[1])] 
                                   for j in range(rho0.shape_0th[0])])
        _out_data_1st = np.array([[(S @ rho0[j,k].data_1st + d)
                                   for k in range(rho0.shape_0th[1])] 
                                   for j in range(rho0.shape_0th[0])])

    return QGstate(data_2nd = _out_data_2nd,
                   data_1st = _out_data_1st,
                   data_0th = rho0.data_0th,
                   dims_fls = rho0.dims_fls,
                   dims_cvs = rho0.dims_cvs
                   )    


def lindblad_evolve(rho0: QGstate, 
                    L: QGsuper, 
                    t = 1
                    ) -> QGstate:
    """
    ---- Prodcedure ----
    Calculate future state of initial state under evolution from a time-independent Lindbladian. The time evolution is
        ρ(t) = exp[-t*L(·)].ρ(0) 
    where L is the Lindbladian, which may be expressed as
        L(ρ) = -i*[H,ρ] + sum_{j,k} Γ_jk (r_k.ρ.r_j - ½*[r_j.r_k,ρ]_+), where H = ½r.H(2).r + r.H(1) + H(0)
    This is equivalent to the following evolution of the moments of ρ(0)
        Σ(t) = S(t).Σ(0).S(t)^T + V(t),   
        μ(t) = S(t).μ(0) + d(t)
    where for the given Lindbladian L, A = Ω.(H(2) + Im[Γ]) and C = -Ω.Re[Γ].Ω:
        S(t) = exp[t*A],    
        d(t) = (1/A).(exp[t*A] - id).(Ω.H(1)),
        V(t) = ∫_0^t exp[(t-s)*A].C.exp[(t-s)*A^T] ds
             = inv_vec( ∫_0^t exp[(t-s)*A]⊗exp[(t-s)*A].vec(C) ds)
             = inv_vec( ∫_0^t exp[(t-s)*A⊕A].vec(C) ds)  where  A⊕B =  A⊗id + id⊗B
             = inv_vec( (1/(A⊕A).(exp[t*A⊕A] - id).vec(C) ).
    In the above "vec" represents the vectorization operation, where "inv_vec" is the inverse function. These are the
    exact solutions to the differential equations for the moments:
        dΣ(t)/dt = A.Σ(t) + Σ(t).A^T + C,
        dμ(t)/dt = A.μ(t) + d(t).

    ---- Parameters ----
    rho0 : QGstate
        Object on which to apply the dynamical transformation generated by the Lindbladian.
    L : QGoper
        Time-independent Lindbladian that will control evolution of the state. Must be CVS only and CPTP.
    t : number
        Time at which to stop calculate the future state. Time starts at t=0, default for the future time is t=1.

    ---- Output ----
    output : QGstate
        The state evolved under the Lindbladian at the give time.
    """
    if L.isfls:
        raise ValueError("Generator of open-system evolution cannot have FLS component.")
    if L.dims_cvs != rho0.dims_cvs:
        raise ValueError("CV component of state and Lindbladian have different dimensions.")
    
    # Construct elements of the open-system transformation
    A = L.wigner_2nd_deriv_var
    C = L.wigner_2nd_deriv
    f = L.wigner_1st_deriv
    S = la.expm(t*A)
    d = t * _exp_integrator_phi_function(t*A) @ f
    V = np.reshape(t * _exp_integrator_phi_function(t*(la.kron(A,np.eye(2*L.dims_cvs)) 
                                                       + la.kron(np.eye(2*L.dims_cvs), A)
                                                       ))
                                                       @ C.ravel(order="C"), L.shape_2nd, order="C")

    if not rho0.isfls:
        _out_data_2nd = S @ rho0.data_2nd @ np.transpose(S) + V
        _out_data_1st = S @ rho0.data_1st + d
    else:
        _out_data_2nd = np.array([[(S @ rho0[j,k].data_2nd @ np.transpose(S) + V) 
                                   for k in range(rho0.shape_0th[1])] 
                                   for j in range(rho0.shape_0th[0])])
        _out_data_1st = np.array([[(S @ rho0[j,k].data_1st + d)
                                   for k in range(rho0.shape_0th[1])] 
                                   for j in range(rho0.shape_0th[0])])

    return QGstate(data_2nd = _out_data_2nd,
                   data_1st = _out_data_1st,
                   data_0th = rho0.data_0th,
                   dims_fls = rho0.dims_fls,
                   dims_cvs = rho0.dims_cvs
                   )