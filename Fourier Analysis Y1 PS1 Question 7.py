##Fourier Analysis Y1 PS1 Question 7
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

def odd(n):
    nums = []
    for i in range(1, 2*n, 2):
        nums.append(i)
    return nums

def triTerm(i, interval):
    x = np.linspace(-interval, interval, 1000)
    nthTerm = 8./np.pi**2/i**2*(-1)**((i-1)/2)*np.sin(i*x)
    return nthTerm

def triFourier(N, interval, scale):
    if scale == "None":
        scale = np.ones(N)

    function = 0
    nList = odd(N)
    for i in range(0,len(nList)):
        function = function + triTerm(nList[i], interval)*scale[i]
    return function

def createScale(N,m):
    scale = np.ones(N)
    for i in range(0, len(scale)):
        if i > m :
            scale[i] = scale[i]/np.sqrt(i-m)
    return scale

def getQuadratureDiff(tri, triScaled, interval=np.pi):
    integrand = (tri - triScaled)**2
    diff = sci.trapezoid(integrand)
    return diff

# %%
#Try with N terms
N = 20
m = 0
x = np.linspace(-np.pi, np.pi, 1000)
scale = createScale(N, m)
tri = triFourier(N, np.pi, scale="None")
triScaled = triFourier(N, np.pi, scale=scale)
plt.plot(x, tri)
plt.plot(x, triScaled)
plt.show()
#%%

mList = []
diffList = []
printCondition=True
for m in range (0, 100):
    scale = createScale(N, m)
    triScaledm = triFourier(N, np.pi, scale=scale)
    diff = getQuadratureDiff(tri, triScaledm)
    diffList.append(diff)
    mList.append(m)
    if diff < 5e-4 and printCondition==True:
        printCondition=False
        print(m)

plt.plot(mList, diffList)
plt.xlabel("m")
plt.ylabel("Quadrature difference")
plt.show()
# %%

# %%
