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

def eavluate_jacobian(J,x,y,z):
    out = np.zeros(3)
    for i in range(3):
        for j in range(3):
            out[i,j] = J[i][j](x,y,z)
    return out
