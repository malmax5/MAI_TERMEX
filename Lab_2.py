import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

l = 9
m = 0.2
t = sp.Symbol('t')
a = sp.cos(t) * math.pi / 3
p1x = l * sp.sin(a) - m / 2 * sp.cos(a)
p2x = l * sp.sin(a) + m / 2 * sp.cos(a)
p3x = m / 2 * sp.cos(a)
p4x = -m / 2 * sp.cos(a)
p1y = l - l * sp.cos(a) - m / 2 * sp.sin(a)
p2y = l - l * sp.cos(a) + m / 2 * sp.sin(a)
p3y = l + m / 2 * sp.sin(a)
p4y = l - m / 2 * sp.sin(a)

h = 0.2
d = 1
s = l / 2 + sp.cos(2 * t + math.pi / 2) * (l / 2 - 2)
x = s * sp.sin(a)
y = l - s * sp.cos(a)
vs = sp.diff(s, t)
vxs = vs * sp.sin(a)
vys = -vs * sp.cos(a)
w = sp.diff(a, t)
vx = w * sp.cos(a) * s
vy = w * sp.sin(a) * s
vpr = sp.sqrt((vxs + vx) * (vxs + vx) + (vys + vy) * (vys + vy))
vps = sp.sqrt(vxs * vxs + vys * vys)
vp = sp.sqrt(vx * vx + vy * vy)


def cir(xl, yl, r):
    angle = np.linspace(0, 2 * np.pi, 150)
    px = xl + r * np.cos(angle)
    py = yl + r * np.sin(angle)
    return px, py


def pr(x1, y1, l, h, a):
    angle = np.linspace(0, 10 * np.pi, 150)
    x1p = np.linspace(0, x1 - h / 2 * np.sin(a), 150)
    y1p = np.linspace(l, y1 + h / 2 * np.cos(a), 150)
    x = x1p + np.sin(angle) * 0.5 * np.cos(a)
    y = y1p + np.sin(angle) * 0.5 * np.sin(a)
    return x, y


def ring(xl, yl, l, h, a):
    Mx = xl + l / 2 * np.cos(a)
    My = yl + l / 2 * np.sin(a)
    Nx = xl - l / 2 * np.cos(a)
    Ny = yl - l / 2 * np.sin(a)
    PX = [Mx - h / 2 * np.sin(a), Mx + h / 2 * np.sin(a), Nx + h / 2 * np.sin(a), Nx - h / 2 * np.sin(a),
          Mx - h / 2 * np.sin(a)]
    PY = [My + h / 2 * np.cos(a), My - h / 2 * np.cos(a), Ny - h / 2 * np.cos(a), Ny + h / 2 * np.cos(a),
          My + h / 2 * np.cos(a)]
    return PX, PY


def rotate(x, y, a):
    x_rotated = x * np.cos(a) - y * np.sin(a)
    y_rotated = x * np.sin(a) + y * np.cos(a)
    return x_rotated, y_rotated


tn = np.linspace(0, 10, 1001)
an = np.zeros_like(tn)

p2xn = np.zeros_like(tn)
p2yn = np.zeros_like(tn)
p4xn = np.zeros_like(tn)
p4yn = np.zeros_like(tn)
p1xn = np.zeros_like(tn)
p1yn = np.zeros_like(tn)
p3xn = np.zeros_like(tn)
p3yn = np.zeros_like(tn)
xn = np.zeros_like(tn)
yn = np.zeros_like(tn)
vxsn = np.zeros_like(tn)
vysn = np.zeros_like(tn)
vxn = np.zeros_like(tn)
vyn = np.zeros_like(tn)
vs = np.zeros_like(tn)
v = np.zeros_like(tn)
vr = np.zeros_like(tn)

for i in range(len(tn)):
    an[i] = sp.Subs(a, t, tn[i])

    p2xn[i] = sp.Subs(p2x, t, tn[i])
    p2yn[i] = sp.Subs(p2y, t, tn[i])
    p4xn[i] = sp.Subs(p4x, t, tn[i])
    p4yn[i] = sp.Subs(p4y, t, tn[i])
    p1xn[i] = sp.Subs(p1x, t, tn[i])
    p1yn[i] = sp.Subs(p1y, t, tn[i])
    p3xn[i] = sp.Subs(p3x, t, tn[i])
    p3yn[i] = sp.Subs(p3y, t, tn[i])
    xn[i] = sp.Subs(x, t, tn[i])
    yn[i] = sp.Subs(y, t, tn[i])
    vxsn[i] = sp.Subs(vxs, t, tn[i])
    vysn[i] = sp.Subs(vys, t, tn[i])
    vxn[i] = sp.Subs(vx, t, tn[i])
    vyn[i] = sp.Subs(vy, t, tn[i])
    v[i] = sp.Subs(vp, t, tn[i])
    vs[i] = sp.Subs(vps, t, tn[i])
    vr[i] = sp.Subs(vpr, t, tn[i])

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(tn, vxsn, color='blue')
ax_for_graphs.set_title("Vxs(t) - stick speed")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(tn, vysn, color='red')
ax_for_graphs.set_title('Vys(t) - stick speed')
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(tn, vxn, color='green')
ax_for_graphs.set_title("Vx(t) - ring speed")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(tn, vyn, color='black')
ax_for_graphs.set_title('Vy(t) - ring speed')
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

fig = plt.figure(figsize=[13, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set_xlim([-l - 5, l + 5])
ax.set(ylim=[-3, m + l + 2])
cx, cy = cir(0, l, m / 2)
ax.plot(cx, cy, 'black')

p1s = ax.plot([p1xn[0], p2xn[0]], [p1yn[0], p2yn[0]], 'black')[0]
p2s = ax.plot([p2xn[0], p3xn[0]], [p2yn[0], p3yn[0]], 'black')[0]
p4s = ax.plot([p4xn[0], p1xn[0]], [p4yn[0], p1yn[0]], 'black')[0]
RX, RY = ring(xn[0], yn[0], d, h, an[0])
rs = ax.plot(RX, RY, 'red')[0]
htr = 0.5
tr = ax.plot([0, -htr, htr, 0], [l, l + htr, l + htr, l], 'black')
cx, cy = pr(xn[0], yn[0], l, h, an[0])
cc, = ax.plot(cx, cy, 'blue')

vvs = ax.plot([xn[0], xn[0] + vxsn[0]], [yn[0], yn[0] + vysn[0]], 'green')[0]
vv = ax.plot([xn[0], xn[0] + vxn[0]], [yn[0], yn[0] + vyn[0]], 'y')[0]
vres = ax.plot([xn[0], xn[0] + vxn[0] + vxsn[0]], [yn[0], yn[0] + vyn[0] + vysn[0]], 'm')[0]

Phis = math.atan2(vysn[0], vxsn[0])
V_arrow_x = np.array([-vs[0] * 0.1, 0.0, -vs[0] * 0.1], dtype=float)
V_arrow_y = np.array([vs[0] * 0.05, 0.0, -vs[0] * 0.05], dtype=float)
V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phis)
V_arrow, = ax.plot(xn[0] + vxsn[0] + V_arrow_rotx, yn[0] + vysn[0] + V_arrow_roty, color="green")

Phi = math.atan2(vyn[0], vxn[0])
Vk_arrow_x = np.array([-v[0] * 0.1, 0.0, -v[0] * 0.1], dtype=float)
Vk_arrow_y = np.array([v[0] * 0.05, 0.0, -v[0] * 0.05], dtype=float)
Vk_arrow_rotx, Vk_arrow_roty = rotate(Vk_arrow_x, Vk_arrow_y, Phi)
Vk_arrow, = ax.plot(xn[0] + vxn[0] + Vk_arrow_rotx, yn[0] + vyn[0] + Vk_arrow_roty, color="y")

Phir = math.atan2(vysn[0] + vyn[0], vxsn[0] + vxn[0])
Vr_arrow_x = np.array([-vr[0] * 0.1, 0.0, -vr[0] * 0.1], dtype=float)
Vr_arrow_y = np.array([vr[0] * 0.05, 0.0, -vr[0] * 0.05], dtype=float)
Vr_arrow_rotx, Vr_arrow_roty = rotate(Vr_arrow_x, Vr_arrow_y, Phir)
Vr_arrow, = ax.plot(xn[0] + vxsn[0] + vxn[0] + Vr_arrow_rotx, yn[0] + vysn[0] + vyn[0] + Vr_arrow_roty, color="m")


def cha(i):
    p1s.set_data([p1xn[i], p2xn[i]], [p1yn[i], p2yn[i]])
    p2s.set_data([p2xn[i], p3xn[i]], [p2yn[i], p3yn[i]])
    p4s.set_data([p4xn[i], p1xn[i]], [p4yn[i], p1yn[i]])

    RX, RY = ring(xn[i], yn[i], d, h, an[i])
    rs.set_data(RX, RY)
    cx, cy = pr(xn[i], yn[i], l, h, an[i])
    cc.set_data(cx, cy)

    vvs.set_data([xn[i], xn[i] + vxsn[i]], [yn[i], yn[i] + vysn[i]])
    vv.set_data([xn[i], xn[i] + vxn[i]], [yn[i], yn[i] + vyn[i]])
    vres.set_data([xn[i], xn[i] + vxn[i] + vxsn[i]], [yn[i], yn[i] + vyn[i] + vysn[i]])

    Phis = math.atan2(vysn[i], vxsn[i])
    V_arrow_x = np.array([-vs[i] * 0.1, 0.0, -vs[i] * 0.1], dtype=float)
    V_arrow_y = np.array([vs[i] * 0.05, 0.0, -vs[i] * 0.05], dtype=float)
    V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phis)
    V_arrow.set_data(xn[i] + vxsn[i] + V_arrow_rotx, yn[i] + vysn[i] + V_arrow_roty)

    Phi = math.atan2(vyn[i], vxn[i])
    Vk_arrow_x = np.array([-v[i] * 0.1, 0.0, -v[i] * 0.1], dtype=float)
    Vk_arrow_y = np.array([v[i] * 0.05, 0.0, -v[i] * 0.05], dtype=float)
    Vk_arrow_rotx, Vk_arrow_roty = rotate(Vk_arrow_x, Vk_arrow_y, Phi)
    Vk_arrow.set_data(xn[i] + vxn[i] + Vk_arrow_rotx, yn[i] + vyn[i] + Vk_arrow_roty)

    Phir = math.atan2(vysn[i] + vyn[i], vxsn[i] + vxn[i])
    Vr_arrow_x = np.array([-vr[i] * 0.1, 0.0, -vr[i] * 0.1], dtype=float)
    Vr_arrow_y = np.array([vr[i] * 0.05, 0.0, -vr[i] * 0.05], dtype=float)
    Vr_arrow_rotx, Vr_arrow_roty = rotate(Vr_arrow_x, Vr_arrow_y, Phir)
    Vr_arrow.set_data(xn[i] + vxsn[i] + vxn[i] + Vr_arrow_rotx, yn[i] + vysn[i] + vyn[i] + Vr_arrow_roty)
    return [p1s], [p2s], [p4s], [rs], [cc], [vvs]


a = FuncAnimation(fig, cha, frames=len(tn), interval=10)
plt.show()