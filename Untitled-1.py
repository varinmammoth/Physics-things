#%%
import numpy as np
import matplotlib.pyplot as plt
positions = np.array([[2, 1, 3], [1, 3, 2]], dtype=np.float64)
velocities = np.zeros_like(positions)
masses = np.array([1, 3, 2], dtype=np.float64)
charges = np.array([1, -4, 2], dtype=np.float64)
# %%
k = 8.987551e9

def E(x, y, q, r):
    """
    Args:
        x (float): X position(s).
        y (float): Y position(s).
        q (float): Charge(s).
        r (iterable of float): (x, y) position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.
    """

    X = r[0]
    Y = r[1]
        
    Ex = []
    Ey = []
    for i in range(0,len(x)):
        x_i = x[i] - X
        y_i = y[i] - Y
        r_i_cube = (x_i**2 + y_i**2)**(3/2)
        ex = np.sum(k*(q/r_i_cube)*x_i)
        ey = np.sum(k*(q/r_i_cube)*y_i)
        Ex.append(ex)
        Ey.append(ey)

    return np.array([Ex, Ey])

def getUnitVector(x, y):
    r = np.sqrt(x**2 + y**2)
    return x/r, y/r

def plotCharge(q, r):
    X = r[0]
    Y = r[1]
    for i in range(0,len(q)):
        if q[i] > 0:
            plt.plot(X[i], Y[i], 'o', markersize=12, c='red')
            plt.plot(X[i], Y[i], '+', markersize=10, c='black')
        else:
            plt.plot(X[i], Y[i], 'o', markersize=12, c='red')
            plt.plot(X[i], Y[i], '_', markersize=10, c='black')
    return
# %%
"""
Testing it out.
Let's have 3 point charges with charge [1,2,-1] at the positions
(0,0) (0.2,0) (0,0.1).
We want the electric field at the positions
(0.2, 0.1) (0.2, 0.1) (1, 2)
Note: I repeated the point twice just to check.
"""
charge_test = np.array([1e-6,-2e-6,3e-6,-1.5e-6])
charge_position_test = np.array([[0,0.2,0,0.15],[0,0,0.1,0.15]])
x_test = np.array([0.2,0.2,1,2])
y_test = np.array([0.1,0.1,0.1,2])
results = E(x_test, y_test, charge_test, charge_position_test)
"""
It works. Now can have arbritary array of x and y, eg. from np.linspace.
"""
# %%
a = 20
x,y = np.meshgrid(np.linspace(0,0.4,a),np.linspace(0,0.4,a))
x = np.reshape(x, (a**2,1))
y = np.reshape(y, (a**2,1))
Ex, Ey = E(x, y, charge_test, charge_position_test)
Ex, Ey = getUnitVector(Ex, Ey)
plt.quiver(x, y, Ex, Ey)
plotCharge(charge_test, charge_position_test)
plt.show()
# %%
#Dipole:
x,y = np.meshgrid(np.linspace(-1.5,1.5,a),np.linspace(-1.5,1.5,a))
x = np.reshape(x, (a**2,1))
y = np.reshape(y, (a**2,1))
charge=np.array([1,-1])
position=np.array([[-1,1],[0,0]])
Ex, Ey = E(x, y, charge, position)
Ex, Ey = getUnitVector(Ex, Ey)
plt.quiver(x, y, Ex, Ey)
plotCharge(charge, position)
plt.show()
# %%
