import numpy as np


def differentiate(u, dt):
    d_n = np.zeros(len(u))
    d_n[0]= (u[1] - u[0]) / dt
    for i in range(1, len(u)-1):
        d_n[i]=(u[i+1]-u[i-1])/(2*dt)
    d_n[-1] = (u[-1] - u[-2]) / dt    
    return d_n

def differentiate_vector(u, dt):
    d= np.zeros(len(u))
    d[0]= (u[1] - u[0]) / dt
    d[1:-1] = (u[2:] -u[0:-2])/(2*dt)
    d[-1] = (u[-1] - u[-2]) / dt
    return d

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1,du2)

if __name__ == '__main__':
    test_differentiate()
    