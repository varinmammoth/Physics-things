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
        x (list of float): X position(s).
        y (list of float): Y position(s).
        q (float): Charge(s).
        r (iterable of float): (x, y) position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.
    
    Returns:
        np.array([Ex, Ey]): array of X and Y components of E field
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
#Turn the block of code above into a single function:
def plotE(xmin,xmax,ymin,ymax,charge,position,a):
    """
    Args:
        xmin (float): minimum x value of chart.
        xmax (float): maximum x value of chart.
        ymin...
        ymax...
        charge (array of float): array of charges
        position (iterable of float): (x, y) position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.
        a: number of vectors in each row and column
        
    """
    x,y = np.meshgrid(np.linspace(xmin,xmax,a),np.linspace(ymin,ymax,a))
    x = np.reshape(x, (a**2,1))
    y = np.reshape(y, (a**2,1))
    Ex, Ey = E(x, y, charge, position)
    Ex, Ey = getUnitVector(Ex, Ey)
    plt.quiver(x, y, Ex, Ey)
    plotCharge(charge, position)
    plt.show()
    return
# %%
#Get force on charge i
#i feels E of everyone else except itself
def getForce(charge, position):
    """
    Args:
        charge (array float): array of charges
        position (iterable of float): (x, y) position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.
    Returns:
        force (array [[fx], [fy]]) force[0] is x components of force on each charge, 
        force[1] is y components of foce on each charge
    """
    Fx = []
    Fy = []
    for i in range(0,len(charge)):
        charge_on_i = charge.tolist()
        x_on_i = position[0].tolist()
        y_on_i = position[1].tolist()
        charge_i = charge_on_i.pop(i)
        x_i = x_on_i.pop(i)
        y_i = y_on_i.pop(i)
        #Note: the .pop(i) command removes ith element and returns it
        Ex_on_i, Ey_on_i = E([x_i], [y_i], charge_on_i, np.array([x_on_i, y_on_i]))
        Fx_on_i, Fy_on_i = charge_i*Ex_on_i[0], charge_i*Ey_on_i[0]
        #Note: Needed the [0] at the end of Ex_on_i and Ey_on_i because
        #generally E(x,y,...) returns an array when x and y are arrays,
        #but here, x and y is just a single-valued array
        Fx.append(Fx_on_i)
        Fy.append(Fy_on_i)
    return np.array([Fx, Fy])
# %%
#Testing the getForce function
testForceCharge = np.array([-1,1,-1])
testForcePosition = np.array([[-1,0,1],[0,0,0]])
plotE(-1,1,-1,1,testForceCharge,testForcePosition,10)
getForce(testForceCharge,testForcePosition)
# %%
#Now, having the force, we can get velocity (dr/dt) and acceleration (d2r/dt2) for the particles
#at times t

def getFuturePos(charge, mass, position, velocity, dt):
    """
    Args:
        charge (array float): array of charges
        mass (array of float): array of masses
        position (iterable of float): (x, y) position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.
        velocity (iterable of float): similiar to position
        dt (float): time increment
    Returns:
        positionNew: array of updated postion
        velocityNew: array of updated velocity
        acceleration: current acceleration, in case we want to plot it
    """
    forces_x, forces_y = getForce(charge, position)
    X, Y = np.array(position[0]), np.array(position[1])
    Vx, Vy = np.array(velocity[0]), np.array(velocity[1])
    Ax, Ay = np.array(np.divide(forces_x, mass)), np.array(np.divide(forces_y, mass))
    Vxnew = Vx+dt*Ax
    Vynew = Vy+dt*Ay
    Xnew = X+dt*Vx
    Ynew = Y+dt*Vy
    positionNew = np.array([Xnew, Ynew])
    velocityNew = np.array([Vxnew, Vynew])
    acceleration = np.array([Ax, Ay])
    return positionNew, velocityNew, acceleration

def simulateCharges(charge, mass, position, velocity, time, dt):
    """
    Simulate motion of charges using finite differences approximation.
    Args:
        charge (array float): array of charges
        mass (array of float): array of masses
        position (iterable of float): (x, y) initial position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.
        velocity (iterable of float): initial velocity. similiar to initial position
        time (float): amount of time we want to simulate. e.g. time=5s will simulate the 
            system for t=0s to t=5s
        dt (float): time increment
    Returns:
        tlist: array of times
        position_t: array of positions at each time in tlist
            Position of charges at time tlist[i] is given by position_t[i]
            position_t[i][0] gives x coordinates of charges at time tlist[i]
            position_t[i][1] gives y coordinates of charges at time tlist[i]
        velocity_t: array of velocity at each time in tlist
            works similiar to postion_t
        acceleration_t: acceleration at each time in tlist. Subtle note: the acceleration
            lags behind by one index. Eg. acceleration_t[i] gives the acceleration at time
            tlist[i-1].
            Works similiar to acceleration_t
    """
    t = 0
    tlist =[]
    position_t = []
    velocity_t = []
    acceleration_t = []
    while t < time:
        tlist.append(t)
        t += dt
        position, velocity, acceleration = getFuturePos(charge, mass, position, velocity, dt)
        position_t.append(position)
        velocity_t.append(velocity)
        acceleration_t.append(acceleration)
    return tlist, position_t, velocity_t, acceleration_t

def getParticleInfo(particleNum, attribute_t):
    particleNum = particleNum - 1 #because particle indexing starts at 0 in python
    x_t = []
    y_t = []
    for i in attribute_t:
        x_t.append(i[0][particleNum])
        y_t.append(i[1][particleNum])
    return x_t, y_t

def animate(tlist, position_t, charge, dt):
    N = len(charge)
    for i in range(0, N):
        exec(f'x{i}_t, y{i}_t = getParticleInfo({i+1}, position_t)')
    
    for i in range(0,len(tlist)):
        plt.clf()
        position_x = []
        position_y = []
        for j in range(0, N):
            exec(f'position_x.append(x{j}_t[i])')
            exec(f'position_y.append(y{j}_t[i])')
        position = np.array([position_x, position_y])
        x,y = np.meshgrid(np.linspace(-1000,1000,a),np.linspace(-1000,1000,a))
        x = np.reshape(x, (a**2,1))
        y = np.reshape(y, (a**2,1))
        Ex, Ey = E(x, y, charge, position)
        Ex, Ey = getUnitVector(Ex, Ey)
        plt.quiver(x, y, Ex, Ey)
        X = position[0]
        Y = position[1]
        for i in range(0,len(charge)):
            if charge[i] > 0:
                plt.plot(X[i], Y[i], 'o', markersize=12, c='red')
                plt.plot(X[i], Y[i], '+', markersize=10, c='black')
            else:
                plt.plot(X[i], Y[i], 'o', markersize=12, c='red')
                plt.plot(X[i], Y[i], '_', markersize=10, c='black')
        plt.pause(dt)
        
# %%
charge = np.array([-3e-19, -3e-19, -3e-19])
position = np.array([[1,2,5], [1,-1,2]])
mass = np.array([1e-31, 1e-31, 1e-31])
velocity = np.array([[0,0,0], [0,0,0]])
time = 5
dt = 0.1
tlist, position_t, velocity_t, acceleration_t = simulateCharges(charge, mass, position, velocity, time, dt)
animate(tlist, position_t, charge, dt)
# %%
