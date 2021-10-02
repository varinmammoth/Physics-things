#%%
import numpy as np
positions = np.array([[2, 1, 3], [1, 3, 2]], dtype=np.float64)
velocities = np.zeros_like(positions)
masses = np.array([1, 3, 2], dtype=np.float64)
charges = np.array([1, -4, 2], dtype=np.float64)
# %%
k = 8.987551e9

def sumArray(a):
    b = np.zeros(len(a))
    for i in range(0,len(a)):
        b[i] = np.sum(a[i])
    return b

def E(x, y, q, r):
    X = r[0]
    Y = r[1]

    # onCharge = False
    # for i in range(0,len(X)):
    #     for j in range(0,len(x)):
    #         if x[j] == X[i] and y[j] == Y[i]:
    #             onCharge = True
            
    # if onCharge == True:
    #         if q[i] > 0:
    #             Ex = np.inf
    #             Ey = np.inf
    #         else:
    #             Ex = -np.inf
    #             Ey = -np.inf
    # else:
        # rcube = ((X-x)**2 + (Y-y)**2)**3
        # Ex = sumArray((k*q/rcube)*x)
        # Ey = sumArray((k*q/rcube)*y)
        
    rcube = ((X-x)**2 + (Y-y)**2)**3
    Ex = sumArray((k*q/rcube)*x)
    Ey = sumArray((k*q/rcube)*y)
    E = np.array([Ex, Ey])
    return E
# %%
