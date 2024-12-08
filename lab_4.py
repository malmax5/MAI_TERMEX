import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


#y[2] = ds
#y[3] = dphi
#y[1] = phi
#y[0] = s
def odesys11(y, t, m, g, P, c, l):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = 1
    a12 = 0
    a21 = 0
    a22 = l+y[0]

    b1 = g*np.cos(y[1])+(l+y[0]-P/c)*y[3]**2-c*g/P*(y[0]-P/c)-m*g/P*y[2]
    b2 = -y[2]*y[3]-g*(l+y[0]-P/c)*np.sin(y[1])

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy



m = 1
g = 9.81
P = 10
c = 10
lens = 9

t_fin = 20

t=np.linspace(0,t_fin,1001)

s0 = 0
phi0 = 0.3
ds0 = 0
dphi0 = 0
y0 = [s0, phi0, ds0, dphi0]

Y11 = odeint(odesys11, y0, t, (m, g, P, c, lens))

s11 = Y11[:, 0]
phi11 = Y11[:, 1]
ds11 = Y11[:, 2]
dphi11 = Y11[:, 3]

fig_for_graphs = plt.figure(figsize=[13,7])
ax_for_graphs = fig_for_graphs.add_subplot(2,2,1)
ax_for_graphs.plot(t,s11,color='blue')
ax_for_graphs.set_title("s(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,2)
ax_for_graphs.plot(t,phi11,color='red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,3)
ax_for_graphs.plot(t,ds11,color='green')
ax_for_graphs.set_title("s'(t)")
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,4)
ax_for_graphs.plot(t,dphi11,color='black')
ax_for_graphs.set_title('phi\'(t)')
ax_for_graphs.set(xlim=[0,t_fin])
ax_for_graphs.grid(True)


plt.show()