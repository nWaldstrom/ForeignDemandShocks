import numpy as np
import numba as nb
from copy import deepcopy

def broyden_solver_autojac(f, x0, y0=None, tol=1E-9, maxcount=500, backtrack_c=0.5, noisy=True, spec_tol=None):
    """Similar to newton_solver, but solves f(x)=0 using approximate rather than exact Newton direction,
    obtaining approximate Jacobian J=f'(x) from Broyden updating (starting from exact Newton at f'(x0)).

    Backtracks only if error raised by evaluation of f, since improvement criterion no longer guaranteed
    to work for any amount of backtracking if Jacobian not exact.
    """

    x, y = x0, y0
    if y is None:
        y = f(x)

    # initialize J with Newton!
    J = obtain_J(f, x, y)
    for count in range(maxcount):
        if noisy:
            printit(count, x, y)
        if spec_tol is None:
            if np.max(np.abs(y)) < tol:
                return x, y
        else:
            if all(np.abs(y) < spec_tol):
                return x, y            

        dx = np.linalg.solve(J, -y)

        # backtrack at most 29 times
        for bcount in range(30):
            # note: can't test for improvement with Broyden because maybe
            # the function doesn't improve locally in this direction, since
            # J isn't the exact Jacobian
            try:
                ynew = f(x + dx)
            except ValueError:
                if noisy:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                J = broyden_update(J, dx, ynew - y)
                y = ynew
                x += dx
                break
        else:
            raise ValueError('Too many backtracks, maybe bad initial guess?')
    else:
        raise ValueError(f'No convergence after {maxcount} iterations')

def printit(it, x, y, **kwargs):
    """Convenience printing function for noisy iterations"""
    print(f'On iteration {it}')
    print(('x = %.3f' + ',%.3f' * (len(x) - 1)) % tuple(x))
    print(('y = %.3f' + ',%.3f' * (len(y) - 1)) % tuple(y))
    for kw, val in kwargs.items():
        print(f'{kw} = {val:.3f}')
    print('\n')

def obtain_J(f, x, y, h=1E-6):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((nx, ny))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx) - y) / h
    return J

def broyden_update(J, dx, dy):
    """Returns Broyden update to approximate Jacobian J, given that last change in inputs to function
    was dx and led to output change of dy."""
    return J + np.outer(((dy - J @ dx) / np.linalg.norm(dx) ** 2), dx)

@nb.njit
def isclose(x,y):
    return np.abs(x-y) < 1e-8 

def get_dX(varname, model, absvalue=False, scaleval=1.0):
    pathvalue = getattr(model.path,varname)[0,:]   
    ssvalue = getattr(model.ss,varname)
    if absvalue:
        dX = (pathvalue-ssvalue) * scaleval
    else:
        dX = (pathvalue-ssvalue) * scaleval / ssvalue
    return dX

def Get_single_HA_IRF(model, input_, Cvar, scaleval):
    ss_C = getattr(model.ss, Cvar)
    dX   = getattr(model.path, input_)[0,:]
    X_ss = getattr(model.ss, input_)
    X_jac_hh = model.jac_hh[(Cvar,input_)]
    return X_jac_hh @ (dX-X_ss)*scaleval / ss_C * 100   

@nb.njit
def variance(x, pi):
    """Variance of discretized random variable with support x and probability mass function pi."""
    return np.sum(pi * (x - np.sum(pi * x)) ** 2)

def std(x, pi):
    """Standard deviation of discretized random variable with support x and probability mass function pi."""
    return np.sqrt(variance(x, pi))

def print_MPCs(model, do_print=False):
    par = model.par
    #sol = model.sol
    #sim = model.sim
    ss = model.ss
    MPC = np.zeros(ss.D.shape)
    MPC[:,:,:-1] = (ss.c[:,:,1:]-ss.c[:,:,:-1])/( (1+model.ss.r)*par.a_grid[np.newaxis,np.newaxis,1:]-(1+model.ss.r)*par.a_grid[np.newaxis,np.newaxis,:-1])
    MPC[:,:,-1] = MPC[:,:,-2] # assuming constant MPC at end

    MPC_ann = 1-(1-MPC)**4
    mean_MPC = np.sum(MPC*ss.D)   
    mean_MPC_ann = np.sum(MPC_ann*ss.D)
    
    std_MPC = std(MPC_ann.flatten(), ss.D.flatten())
    if do_print: print(f'mean MPC: {mean_MPC:.3f} [annual: {mean_MPC_ann:.3f}], std of ann. MPCs: {std_MPC:.3f}')
    return mean_MPC_ann, std_MPC

def print_MPCs_s(model, i_s, do_print=False):
    par = model.par
    sol = model.sol
    sim = model.sim
    #ss = model.ss
    sdim1 = i_s*par.Ne
    sdim2 = (1+i_s)*par.Ne
    MPC = np.zeros(ss.D[:,sdim1:sdim2,:].shape)
 
    temp = (ss.c[:,sdim1:sdim2,1:]-ss.c[:,sdim1:sdim2,:-1])/( (1+model.ss.r)*par.a_grid[np.newaxis,np.newaxis,1:]-(1+model.ss.r)*par.a_grid[np.newaxis,np.newaxis,:-1])
    
    if i_s == 0 : fac = model.par.sT
    if i_s == 1 : fac = 1-model.par.sT
    
    MPC[:,:,:-1] = temp / fac
    MPC[:,:,-1] = MPC[:,:,-2] # assuming constant MPC at end
    
    MPC_ann = 1-(1-MPC)**4
    mean_MPC = np.sum(MPC*ss.D[:,sdim1:sdim2,:])   
    mean_MPC_ann = np.sum(MPC_ann*ss.D[:,sdim1:sdim2,:])   
    
    std_MPC = std(MPC.flatten(), ss.D[:,sdim1:sdim2,:].flatten())
    if do_print: print(f'mean MPC: {mean_MPC:.3f} [annual: {mean_MPC_ann:.3f}], std of MPCs: {std_MPC:.3f}')
    return mean_MPC_ann, std_MPC    

def Create_shock(var, model, ssize=0.01, absval=False, custompath=None):   
    for shockname in model.shocks:
        patharray = getattr(model.path,shockname)
        ssvalue = getattr(model.ss,shockname)
        patharray[:,:] = ssvalue 
            
    if absval:
        patharray = getattr(model.path,var)
        ssvalue = getattr(model.ss,var)    
        rho = getattr(model.par, 'rho_'+var)
        T_half = model.par.T//2
        patharray[:,:T_half] = ssvalue +  ssize*rho**np.arange(T_half)
        patharray[:,T_half:] = ssvalue  
    elif custompath is not None:
        for var_ in custompath:
            shockpath = getattr(model.ss, var_) + custompath[var_]
            patharray = getattr(model.path,var_)
            patharray[:,:] = shockpath
    else:
        patharray = getattr(model.path,var)
        ssvalue = getattr(model.ss,var)    
        rho = getattr(model.par, 'rho_'+var)
        T_half = model.par.T//2
        patharray[:,:T_half] = ssvalue +  ssvalue*ssize*rho**np.arange(T_half)
        patharray[:,T_half:] = ssvalue  



def scaleshock(var, model, size=0.01, abs=False, cumeffect=False, Nq=8, variance=False, sign=1):
    if cumeffect:
        InitShock = np.sum(getattr(model.path,var)[0,:Nq] / getattr(model.ss,var) -1)
        model.par.scale = size / InitShock     
    elif variance:
        dLog = np.log(getattr(model.path,var)[0,:]) - np.log(getattr(model.ss,var))
        InitShock_var = np.sum(dLog**2)
        model.par.scale = sign*np.sqrt(size / InitShock_var) 

        test = (np.log(getattr(model.path,var)[0,:]) - np.log(getattr(model.ss,var)))* model.par.scale
        assert np.isclose(np.sum(test**2), size)      
    elif not abs:
        InitShock = getattr(model.path,var)[0,0] / getattr(model.ss,var) -1
        model.par.scale = size / InitShock        
    else:
        InitShock = getattr(model.path,var)[0,0] - getattr(model.ss,var) 
        model.par.scale = size / InitShock        
 

def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

   
def sim_earnings(rho_e, sigma_e, T, N):
    LogE = np.zeros([T,N])
    LogE[0,:] = 0.0    
    for t in range(T-1):
        LogE[t+1,:] = rho_e*LogE[t,:] + np.random.normal(loc=0.0, scale=sigma_e,size=N) 
    return LogE

def markov_rouwenhorst(rho, sigma, N=7):
    """Rouwenhorst method analog to markov_tauchen"""

    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2

    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma     / np.sqrt(variance(s, pi)))
    y = np.exp(s) / np.sum(pi * np.exp(s))

    return y, pi, Pi

def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed

    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi


def nonlin_MPC(model):
    model._set_inputs_hh_all_ss()
    
    prop_shock = False
    
    model.path.UniformT[:,:] = 0.0 
    model.path.UniformT[0,0] = 0.05 * (model.ss.WT * model.ss.NT + model.ss.WNT * model.ss.NNT) #  500$
    dI = model.path.UniformT[0,0]      
    model.solve_hh_path()
    model.simulate_hh_path()
    
      

    dCAgg = np.zeros(model.par.T)*np.nan
    dCAgg_T = np.zeros(model.par.T)*np.nan
    dCAgg_NT = np.zeros(model.par.T)*np.nan
    C_ss = np.sum(model.ss.c*model.ss.D)
    
    for t in range(model.par.T):
        if prop_shock:
            term1 = np.sum(((model.path.c[t]-model.ss.c)/dI[np.newaxis,:,np.newaxis]) * model.ss.D)
            term2 = np.sum(model.ss.c * (model.path.D[t] - model.ss.D)/dI[np.newaxis,:,np.newaxis]) 
        else:
            term1 = (np.sum(model.path.c[t]*model.path.D[t])-C_ss)/dI
            term2 = 0
        mpc_i = term1 + term2 
        dCAgg[t] = deepcopy(mpc_i) 

    for t in range(model.par.T):
        if prop_shock:
            term1 = np.sum(((model.path.c_T[t]-model.ss.c_T)/dI[np.newaxis,:,np.newaxis]) * model.ss.D)
            term2 = np.sum(model.ss.c_T * (model.path.D[t] - model.ss.D)/dI[np.newaxis,:,np.newaxis]) 
        else:
            term1 = np.sum(((model.path.c_T[t]-model.ss.c_T)/dI) * model.ss.D)
            term2 = np.sum(model.ss.c_T * (model.path.D[t] - model.ss.D)/dI)             
        mpc_i = term1 + term2 
        dCAgg_T[t] = deepcopy(mpc_i) / model.par.sT

    for t in range(model.par.T):
        if prop_shock:
            term1 = np.sum(((model.path.c_NT[t]-model.ss.c_NT)/dI[np.newaxis,:,np.newaxis]) * model.ss.D)
            term2 = np.sum(model.ss.c_NT * (model.path.D[t] - model.ss.D)/dI[np.newaxis,:,np.newaxis]) 
        else:
            term1 = np.sum(((model.path.c_NT[t]-model.ss.c_NT)/dI) * model.ss.D)
            term2 = np.sum(model.ss.c_NT * (model.path.D[t] - model.ss.D)/dI)             
        mpc_i = term1 + term2 
        dCAgg_NT[t] = deepcopy(mpc_i) / (1-model.par.sT)
        
    ann_dCAgg = np.zeros(round(model.par.T/4))
    ann_dCAgg_T = np.zeros(round(model.par.T/4))
    ann_dCAgg_NT = np.zeros(round(model.par.T/4))
    for j in range(round(model.par.T/4)):
        ann_dCAgg[j] = np.sum(dCAgg[j*4:(1+j)*4])  
        ann_dCAgg_T[j] = np.sum(dCAgg_T[j*4:(1+j)*4])  
        ann_dCAgg_NT[j] = np.sum(dCAgg_NT[j*4:(1+j)*4])  
    
    model._set_inputs_hh_all_ss()
    return ann_dCAgg, dCAgg, ann_dCAgg_T, ann_dCAgg_NT

def homotopic_cont(model, target_vars, target_vals, nsteps = 3, noisy=True, orgvals=None):
    if orgvals is None:
        orgvals = [getattr(model.par,x) for x in target_vars]
    stpsize = 0
    for j in range(nsteps):             
        updatedvals = [orgvals[k] * (1- (1+stpsize+j)/(stpsize+nsteps)) + target_vals[k] * (1+stpsize+j)/(stpsize+nsteps) for k in range(len(target_vars))]
        if noisy: print(f'Iteration {1+j}')
        for k in range(len(target_vars)):
            setattr(model.par, target_vars[k], updatedvals[k])
            if noisy: print(f'Parameter value for {target_vars[k]} = {updatedvals[k]:.3f}')
       
        model.find_ss(do_print=False)



def get_GE_mat(model, totvarlist=None, shocks_exo=None, calc_jac=True):
    
    T=model.par.T

    if totvarlist is None: totvarlist = model.varlist
    if shocks_exo is None: shocks_exo = model.shocks

    if calc_jac:
        if model.par.HH_type == 'HA':
            model.compute_jacs(do_print=False,skip_shocks=False)
        else:
            model.compute_jacs(do_print=False,skip_shocks=False,skip_hh=True)

    # unknowns
    model.G_U[:,:] = -np.linalg.solve(model.H_U,model.H_Z)  
    dU = model.G_U  
    finjac = model.jac.copy()

    for i_shock,shockname in enumerate(model.shocks):
        for varname in totvarlist:
            finjac[(varname,shockname)] = model.jac[(varname,shockname)]        
            for i_input,inputname in enumerate(model.unknowns):
                finjac[(varname,shockname)] += model.jac[(varname,inputname)] @ model.G_U[i_input*T:(1+i_input)*T,i_shock*T:(1+i_shock)*T] 
                
    return finjac 

