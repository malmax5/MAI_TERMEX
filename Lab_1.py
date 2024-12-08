import numpy as np
import matplotlib.pyplot as plt
import math

# Исходные параметры
v0 = 10
R = 5
T = np.linspace(0, 20, 2001)  # Время от 0 до 20 с шагом 0.01

# Вычисления траектории
rx = v0 * T - R * np.cos(v0 * T / R - math.pi / 2)
ry = R + R * np.sin(v0 * T / R - math.pi / 2)

# Скорости по осям
vpx = v0 - (R * np.sin(v0 * T / R - math.pi / 2) * v0 / R)
vpy = R * np.cos(v0 * T / R - math.pi / 2) * v0 / R

# Модуль скорости
vp = np.sqrt(vpx**2 + vpy**2)

# Угловые ускорения
wpx = np.gradient(vpx, T)
wpy = np.gradient(vpy, T)

# Угловая скорость
Wt = np.gradient(vp, T)
W = np.sqrt(wpx**2 + wpy**2)
Wn = np.sqrt(np.abs(W**2 - Wt**2))

# Начальные значения для графика
xn = rx
yn = ry
vx = vpx
vy = vpy
v = vp
wt = Wt
wn = Wn

# Создание фигуры для анимации
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.plot(xn, yn)

P = ax.plot(xn[0], yn[0], marker='o')[0]

# Начальный угол Phi
Phi = np.arctan2(vy[0], vx[0])

VLine = ax.plot([xn[0], xn[0] + vx[0]], [yn[0], yn[0] + vy[0]], 'black')[0]

# Функция для вращения стрелки
def rotate(x, y, a):
    x_rotated = x * np.cos(a) - y * np.sin(a)
    y_rotated = x * np.sin(a) + y * np.cos(a)
    return x_rotated, y_rotated

# Стрелки скорости и ускорения
V_arrow_x = np.array([-v[0]*0.1, 0.0, -v[0]*0.1], dtype=float)
V_arrow_y = np.array([v[0]*0.05, 0.0, -v[0]*0.05], dtype=float)
V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phi)
V_arrow, = ax.plot(xn[0] + vx[0] + V_arrow_rotx, yn[0] + vy[0] + V_arrow_roty, color="black")

WTLine = ax.plot([xn[0], xn[0] + wt[0]*math.cos(Phi)], [yn[0], yn[0] + wt[0]*math.sin(Phi)], 'red')[0]

WT_arrow_x = np.array([-wt[0]*0.1, 0.0, -wt[0]*0.1], dtype=float)
WT_arrow_y = np.array([wt[0]*0.05, 0.0, -wt[0]*0.05], dtype=float)
WT_arrow_rotx, WT_arrow_roty = rotate(WT_arrow_x, WT_arrow_y, Phi)
WT_arrow, = ax.plot(xn[0] + wt[0]*math.cos(Phi) + WT_arrow_rotx, yn[0] + wt[0]*math.sin(Phi) + WT_arrow_roty, color="red")

WNLine = ax.plot([xn[0], xn[0] + wn[0]*math.cos(Phi - math.pi/2)], [yn[0], yn[0] + wn[0]*math.sin(Phi - math.pi/2)], 'blue')[0]

WN_arrow_x = np.array([-wn[0]*0.1, 0.0, -wn[0]*0.1], dtype=float)
WN_arrow_y = np.array([wn[0]*0.05, 0.0, -wn[0]*0.05], dtype=float)
WN_arrow_rotx, WN_arrow_roty = rotate(WN_arrow_x, WN_arrow_y, Phi - math.pi/2)
WN_arrow, = ax.plot(xn[0] + wn[0]*math.cos(Phi - math.pi/2) + WN_arrow_rotx, yn[0] + wn[0]*math.sin(Phi - math.pi/2) + WN_arrow_roty, color="blue")

# Функция обновления для анимации
def cha(i):
    P.set_data(xn[i], yn[i])

    Phi = np.arctan2(vy[i], vx[i])

    VLine.set_data([xn[i], xn[i] + vx[i]], [yn[i], yn[i] + vy[i]])

    V_arrow_x = np.array([-v[i]*0.1, 0.0, -v[i]*0.1], dtype=float)
    V_arrow_y = np.array([v[i]*0.05, 0.0, -v[i]*0.05], dtype=float)
    V_arrow_rotx, V_arrow_roty = rotate(V_arrow_x, V_arrow_y, Phi)
    V_arrow.set_data(xn[i] + vx[i] + V_arrow_rotx, yn[i] + vy[i] + V_arrow_roty)

    WTLine.set_data([xn[i], xn[i] + wt[i] * math.cos(Phi)], [yn[i], yn[i] + wt[i] * math.sin(Phi)])

    WT_arrow_x = np.array([-wt[i]*0.1, 0.0, -wt[i]*0.1], dtype=float)
    WT_arrow_y = np.array([wt[i]*0.05, 0.0, -wt[i]*0.05], dtype=float)
    WT_arrow_rotx, WT_arrow_roty = rotate(WT_arrow_x, WT_arrow_y, Phi)
    WT_arrow.set_data(xn[i] + wt[i] * math.cos(Phi) + WT_arrow_rotx, yn[i] + wt[i] * math.sin(Phi) + WT_arrow_roty)

    WNLine.set_data([xn[i], xn[i] + wn[i] * math.cos(Phi - math.pi / 2)], [yn[i], yn[i] + wn[i] * math.sin(Phi - math.pi / 2)])

    WN_arrow_x = np.array([-wn[i]*0.1, 0.0, -wn[i]*0.1], dtype=float)
    WN_arrow_y = np.array([wn[i]*0.05, 0.0, -wn[i]*0.05], dtype=float)
    WN_arrow_rotx, WN_arrow_roty = rotate(WN_arrow_x, WN_arrow_y, Phi - math.pi / 2)
    WN_arrow.set_data(xn[i] + wn[i] * math.cos(Phi - math.pi / 2) + WN_arrow_rotx, yn[i] + wn[i] * math.sin(Phi - math.pi / 2) + WN_arrow_roty)

    return [P]

# Создание анимации
from matplotlib.animation import FuncAnimation
a = FuncAnimation(fig, cha, frames=len(T), interval=10)

# Отображение
plt.show()
