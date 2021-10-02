#%%
import numpy as np
import matplotlib.pylot as plt
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
# %%
"""
Testing it out.
Let's have 3 point charges with charge [1,2,-1] at the positions
(0,0) (0.2,0) (0,0.1).
We want the electric field at the positions
(0.2, 0.1) (0.2, 0.1) (1, 2)
Note: I repeated the point twice just to check.
"""
charge_test = np.array([1e-6,-2e-6,3e-6])
charge_position_test = np.array([[0,0.2,0],[0,0,0.1]])
x_test = np.array([0.2,0.2,1])
y_test = np.array([0.1,0.1,0.1])
results = E(x_test, y_test, charge_test, charge_position_test)
"""
It works. Now can have arbritary array of x and y, eg. from np.linspace.
"""
# %%
x = np.linspace(-5,5,10)
y = np.linspace(-5,5,10)
Ex, Ey = E(x, y, charge_test, charge_position_test)
plt.quiver(x, y, Ex, Ey)
plt.show