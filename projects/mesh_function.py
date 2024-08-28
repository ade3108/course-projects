import numpy as np
#from numpy import *
import matplotlib.pyplot as plt
#from matplotlib.pyplot import *


def mesh_function(f, t):
    b = np.zeros(len(t))
    for i, a in enumerate(t):
        b[i] = func(a)
    #plot_mesh_function(f,t,0.1)
    return b


def func(t):
    if t <= 3 and t >= 0:
        return np.exp(-t)
    elif  t > 3 and t<= 4:
        return np.exp(-3*t)
    

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

def plot_mesh_function(f, t):
    N=100
    x=np.linspace(0,t,N)
    y= mesh_function(f, x)

    plt.plot(x,y, 'r--o')
    plt.xlabel("$t$")
    plt.ylabel(r"$\exp(x)$")
    plt.show()

if __name__ == "__main__":
    test_mesh_function()
    t = np.linspace(0, 10, 100)
    plot_mesh_function(func, 10)