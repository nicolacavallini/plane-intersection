import sympy as sy
import numpy as np
from sympy.utilities.lambdify import lambdify

def get_jacobian(f,x,y):
    return (sy.diff(f,x),sy.diff(f,y))

def get_lamdas(f,x,y):
    gx,gy = get_jacobian(f,x,y)
    fl = lambdify((x,y),f)
    gxl = lambdify((x,y),gx)
    gyl = lambdify((x,y),gy)
    return fl, gxl, gyl

def subs_nomals_in_function(f,k,n1,r,n2):
    for i,j in zip(k,n1):
        f = f.subs(i,j)

    for i,j in zip(r,n2):
        f = f.subs(i,j)

    return f

#def eavluate_jacobian(J,x,y,z):
def eavluate_jacobian(J,v):
    out = np.zeros((3,3))
    assert(v.shape==(3,1))
    assert(len(J)==3)
    assert(len(J[0])==3)
    assert(len(J[1])==3)
    assert(len(J[2])==3)
    x = v[0,0]
    y = v[1,0]
    z = v[2,0]
    for i in range(3):
        for j in range(3):
            out[i,j] = J[i][j](x,y,z)
    return out

def eavluate_function(F,v):
    assert(v.shape==(3,1))
    assert(len(F)==3)
    x = v[0,0]
    y = v[1,0]
    z = v[2,0]
    return np.array([[F[0](x,y,z)],[F[1](x,y,z)],[F[2](x,y,z)]])

def newton_solve(F,J,x0,tol=1e-10,iters=100,verbose=True):
    for i in range(iters):
        jac = eavluate_jacobian(J,x0)
        func = eavluate_function(F,x0)
        x1 = x0 - np.linalg.solve(jac,func)
        residual = np.sqrt(np.sum(func**2))
        x0 = x1
        if verbose:
            print("i = ",i,", residual = ",residual)
        if (residual<tol):
            return x1
        if (i==iters-1):
            print("NOT converged")
            return np.array([[np.nan],[np.nan],[np.nan]])
