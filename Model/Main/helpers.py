import numba as nb
import numpy as np
import utils
from GEModelTools import lag, lead


@nb.njit
def chain_price_index(P, ssP, Y, P1, ssP1, Y1, P2, ssP2, Y2, T, sign=1.0):
    for t in range(T):
        if t==0:
            Y[t]  = (ssP1 * Y1[t] + sign*(ssP2 * Y2[t]))/ssP
        else:
            Y[t]  = (P1[t-1] * Y1[t] + sign*(P2[t-1] * Y2[t]))/P[t-1]
        P[t] = (P1[t] * Y1[t] + sign*(P2[t] * Y2[t]))/Y[t]


@nb.njit
def CES_demand(c, PH, PF, P, eta, alpha, c_bar=0.0):
    cF = alpha     * (PF/P)**(-eta) * (c - c_bar*PF/P) + c_bar  
    cH = (1-alpha) * (PH/P)**(-eta) * (c - c_bar*PF/P)
    return cF, cH


@nb.njit
def CES_demand_3_inputs(X, PH, PNT, PF,PX, eta, Theta_T, Theta_NT, Theta_M):
    XT2T  = Theta_T     * (PH/PX)**(-eta) * X 
    XNT2T = Theta_NT * (PNT/PX)**(-eta) * X
    XM2T  = Theta_M * (PF/PX)**(-eta) * X
    return XT2T, XNT2T, XM2T


@nb.njit
def get_intermediates(XT, XNT, PXT, PXNT, PH, PNT, PF, Theta_T, Theta_NT, etaX):  
    XT2T, XNT2T, XM2T = CES_demand_3_inputs(XT, PH, PNT, PF, PXT, etaX, Theta_T[0], Theta_T[1], Theta_T[2]) # intermediate demand in tradeable sector
    XNT2NT, XT2NT = CES_demand(XNT, PH, PNT, PXNT, etaX, Theta_NT[0]) # intermediate demand in non-tradeable sector
    return XT2NT, XNT2T, XNT2NT, XT2T, XM2T

@nb.njit
def CES_demand_T(C, PT, PNT, P, etaT, alphaT, c_bar=0.0):
    CT = alphaT     * (PT/P)**(-etaT) * (C - c_bar*PT/P) + c_bar  
    CNT = (1-alphaT) * (PNT/P)**(-etaT) * (C - c_bar*PNT/P) 
    return CT, CNT

@nb.njit
def Armington(PH_s, PF_s, C_s, gamma, alpha):
    CH_s = alpha * (PH_s/PF_s)**(-gamma) * C_s
    return CH_s

@nb.njit
def Price_index(PH, PF, eta, alpha):
    if utils.isclose(eta,1.0):
        P = PF**alpha * PH**(1-alpha)
    else:
        P = (alpha*PF**(1-eta) + (1-alpha)*PH**(1-eta))**(1/(1-eta))
    return P

@nb.njit
def Price_index_T(PH, PNT, PF, etaX, X_share_T):
    if utils.isclose(etaX,1.0):
        PXT = PH**X_share_T[0] * PNT**X_share_T[1] * PF**X_share_T[2]
    else:
        PXT = (X_share_T[0]*PH**(1-etaX) + X_share_T[1]*PNT**(1-etaX) + X_share_T[2]*PF**(1-etaX))**(1/(1-etaX))
    return PXT

@nb.njit
def Price_index_NT(PH, PNT, etaX, X_share_NT):
    if utils.isclose(etaX,1.0):
        PXNT = PNT**X_share_NT[0] * PH**X_share_NT[1]  
    else:
        PXNT = (X_share_NT[0]*PNT**(1-etaX) + X_share_NT[1]*PH**(1-etaX))**(1/(1-etaX))
    return PXNT

@nb.njit
def sol_Price_index_rel(pF, eta, alpha):
    if utils.isclose(eta,1.0):
        pH = (1.0/(pF**alpha))**(1/(1-alpha))
    else:
        pH = ((1.0-alpha*pF**(1-eta))/(1-alpha))**(1/(1-eta))
    return pH


@nb.njit
def sol_Price_index2(pH, pF, eta, alpha):
    if utils.isclose(eta,1.0):
        pF = (1.0/(pH**(1-alpha)))**(1/alpha)
    else:
        pF = ((1.0-(1-alpha)*pH**(1-eta))/alpha)**(1/(1-eta))
    return pF

@nb.njit
def Inf(P, ssval):
    P_lag = lag(ssval,P) 
    pi = P/P_lag - 1
    pi_p = lead(pi,0)
    return pi, pi_p

@nb.njit
def P_from_inf(P, inf, T, ssP):
    for t in range(T):
        if t == 0:
            P[t] = ssP*(1+inf[t]) 
        else:
            P[t] = P[t-1]*(1+inf[t]) 


@nb.njit
def Get_HH_A(A, C, I, ra, T, ss):
    for t in range(T):  
        if t==0:
            A[t] = ss.A * (1+ra[t]) + I[t]   - C[t] 
        else:
            A[t] = A[t-1] * (1+ra[t]) + I[t]   - C[t]   
            

@nb.njit    
def sol_backwards_lin(var, ssvar, a, b, T):
    for tt in range(T):
        t = T-1-tt
        if t == T-1:
            var[t] = ssvar
        else:
            var[t] = a[t] + b[t] * var[t+1]   
            
def sol_backwards(var, ssvar, par, f, *args):
    for tt in range(par.T):
        t = par.T-1-tt
        if t == par.T-1:
            var[t] = ssvar
        else:
            var[t] = f(t, var[t+1], *args)        

def sol_forwards(var, ssvar, par, f, *args):
    for t in range(par.T):
        if t == 0:
            var[t] = f(t, ssvar, *args)
        else:
            var[t] = f(t, var[t-1], *args)     